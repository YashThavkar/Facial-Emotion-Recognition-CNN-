import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import queue
from collections import deque
import json

class RealTimeEmotionClassifier:
    def __init__(self, camera_index=0, analysis_interval=0.5):
        """
        Initialize real-time emotion classifier
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            analysis_interval (float): Time interval between analyses in seconds
        """
        self.camera_index = camera_index
        self.analysis_interval = analysis_interval
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Threading and queue setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=5)
        self.analysis_thread = None
        self.running = False
        
        # Results tracking
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.emotion_history = deque(maxlen=10)
        self.last_analysis_time = 0
        
        # Colors for emotions (BGR format for OpenCV)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Dark Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255)  # Orange
        }
    
    def analyze_frame_worker(self):
        """
        Worker thread for analyzing frames
        """
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame = self.frame_queue.get(timeout=1.0)
                
                # Analyze emotion
                result = self.analyze_emotion(frame)
                
                # Put result in result queue
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def analyze_emotion(self, frame):
        """
        Analyze emotion in a single frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            dict: Analysis result
        """
        try:
            # Analyze the frame using DeepFace
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Extract emotion data
            if isinstance(result, list):
                emotions = result[0]['emotion']
                region = result[0].get('region', {})
            else:
                emotions = result['emotion']
                region = result.get('region', {})
            
            # Get dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            
            return {
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions,
                'region': region,
                'timestamp': time.time(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {},
                'region': {},
                'timestamp': time.time(),
                'status': f'error: {str(e)}'
            }
    
    def draw_results(self, frame, result):
        """
        Draw emotion analysis results on frame
        
        Args:
            frame: OpenCV frame
            result: Analysis result dictionary
        """
        height, width = frame.shape[:2]
        
        # Draw emotion text
        emotion = result['dominant_emotion']
        confidence = result['confidence']
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Main emotion label
        label = f"{emotion.upper()}: {confidence:.1f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, color, 2, cv2.LINE_AA)
        
        # Draw face region if available
        region = result.get('region', {})
        if region and all(k in region for k in ['x', 'y', 'w', 'h']):
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion bars
        y_offset = 70
        bar_width = 200
        bar_height = 15
        
        cv2.putText(frame, "Emotion Scores:", (10, y_offset - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (emo, score) in enumerate(result['all_emotions'].items()):
            y_pos = y_offset + i * 25
            bar_length = int((score / 100) * bar_width)
            emo_color = self.emotion_colors.get(emo, (255, 255, 255))
            
            # Draw emotion name
            cv2.putText(frame, f"{emo}:", (10, y_pos + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw score bar
            cv2.rectangle(frame, (80, y_pos), (80 + bar_width, y_pos + bar_height), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (80, y_pos), (80 + bar_length, y_pos + bar_height), 
                         emo_color, -1)
            
            # Draw score text
            cv2.putText(frame, f"{score:.1f}%", (290, y_pos + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw status and FPS
        status_color = (0, 255, 0) if result['status'] == 'success' else (0, 0, 255)
        cv2.putText(frame, f"Status: {result['status']}", (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def update_emotion_history(self, result):
        """
        Update emotion history for smoothing
        
        Args:
            result: Analysis result dictionary
        """
        if result['status'] == 'success':
            self.emotion_history.append({
                'emotion': result['dominant_emotion'],
                'confidence': result['confidence'],
                'timestamp': result['timestamp']
            })
            
            # Update current emotion (could implement smoothing here)
            self.current_emotion = result['dominant_emotion']
            self.current_confidence = result['confidence']
    
    def get_emotion_statistics(self):
        """
        Get statistics from emotion history
        
        Returns:
            dict: Emotion statistics
        """
        if not self.emotion_history:
            return {}
        
        # Count emotions in history
        emotion_counts = {}
        total_confidence = 0
        
        for entry in self.emotion_history:
            emotion = entry['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += entry['confidence']
        
        # Calculate percentages
        total_entries = len(self.emotion_history)
        emotion_percentages = {
            emotion: (count / total_entries) * 100
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'average_confidence': total_confidence / total_entries,
            'total_samples': total_entries
        }
    
    def save_screenshot(self, frame, result):
        """
        Save screenshot with emotion analysis
        
        Args:
            frame: OpenCV frame
            result: Analysis result
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        emotion = result['dominant_emotion']
        filename = f"emotion_capture_{emotion}_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        
        # Also save analysis result
        json_filename = f"emotion_data_{emotion}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Analysis data saved: {json_filename}")
    
    def run(self):
        """
        Main loop for real-time emotion classification
        """
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting real-time emotion classification...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        self.running = True
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analyze_frame_worker)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        # Default result for initial display
        current_result = {
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'all_emotions': {emotion: 0.0 for emotion in self.emotions},
            'region': {},
            'status': 'initializing'
        }
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 0)
            
                
                # Add frame to analysis queue if it's time and queue isn't full
                current_time = time.time()
                if (current_time - self.last_analysis_time >= self.analysis_interval and 
                    not self.frame_queue.full()):
                    self.frame_queue.put(frame.copy())
                    self.last_analysis_time = current_time
                
                # Get latest analysis result if available
                try:
                    while not self.result_queue.empty():
                        current_result = self.result_queue.get_nowait()
                        self.update_emotion_history(current_result)
                except queue.Empty:
                    pass
                
                # Draw results on frame
                self.draw_results(frame, current_result)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    start_time = time.time()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Real-time Emotion Classification', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame, current_result)
                elif key == ord('h'):
                    # Print emotion history statistics
                    stats = self.get_emotion_statistics()
                    print("\nEmotion History Statistics:")
                    print("==========================")
                    for emotion, percentage in stats.get('emotion_percentages', {}).items():
                        print(f"{emotion}: {percentage:.1f}%")
                    print(f"Average confidence: {stats.get('average_confidence', 0):.1f}%")
                    print(f"Total samples: {stats.get('total_samples', 0)}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            
            if self.analysis_thread:
                self.analysis_thread.join(timeout=2)
            
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\nFinal Emotion Statistics:")
            print("========================")
            stats = self.get_emotion_statistics()
            for emotion, percentage in stats.get('emotion_percentages', {}).items():
                print(f"{emotion}: {percentage:.1f}%")

def main():
    print("Real-time Emotion Classification")
    print("================================")
    
    # Get camera index
    camera_index = input("Enter camera index (0 for default, press Enter for 0): ").strip()
    camera_index = int(camera_index) if camera_index.isdigit() else 0
    
    # Get analysis interval
    interval = input("Enter analysis interval in seconds (0.5 for default): ").strip()
    try:
        interval = float(interval) if interval else 0.5
    except ValueError:
        interval = 0.5
    
    # Initialize and run classifier
    classifier = RealTimeEmotionClassifier(camera_index=camera_index, 
                                         analysis_interval=interval)
    
    print(f"\nUsing camera index: {camera_index}")
    print(f"Analysis interval: {interval} seconds")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press 'h' to print emotion history statistics")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    classifier.run()

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import deepface
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "deepface", "opencv-python"])
        import deepface
    
    main()