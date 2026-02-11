import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template_string, Response
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetectionModel:
    def __init__(self, model_path, input_size=(48, 48)):
        """
        Initialize emotion detection model
        
        Args:
            model_path (str): Path to the .h5, .keras, or .h5.keras model file
            input_size (tuple): Expected input size for the model
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
        ]
        self.emotion_emojis = {
            'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
            'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()
    
    def load_model(self):
        """Load the trained emotion detection model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model based on file extension
            if self.model_path.endswith(('.h5', '.keras', '.h5.keras')):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Successfully loaded model from {self.model_path}")
                logger.info(f"Model input shape: {self.model.input_shape}")
                logger.info(f"Model output shape: {self.model.output_shape}")
            else:
                raise ValueError("Unsupported model format. Use .h5, .keras, or .h5.keras files")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_face(self, face_roi):
        """
        Preprocess face region for emotion prediction
        
        Args:
            face_roi (numpy.ndarray): Face region of interest
            
        Returns:
            numpy.ndarray: Preprocessed face data
        """
        try:
            # Convert to grayscale if needed
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            face_roi = cv2.resize(face_roi, self.input_size)
            
            # Normalize pixel values
            face_roi = face_roi.astype('float32') / 255.0
            
            # Reshape for model input (add batch and channel dimensions)
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            return face_roi
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def detect_faces(self, image):
        """
        Detect faces in the image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of face coordinates (x, y, w, h)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def predict_emotion(self, image):
        """
        Predict emotion from image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Emotion prediction results
        """
        try:
            if self.model is None:
                return {"error": "Model not loaded", "success": False}
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                return {"error": "No faces detected in the image", "success": False}
            
            results = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = self.preprocess_face(face_roi)
                
                if processed_face is None:
                    continue
                
                # Predict emotion
                emotion_probs = self.model.predict(processed_face, verbose=0)[0]
                
                # Get emotion scores
                emotion_scores = {
                    label: float(prob * 100) for label, prob in zip(self.emotion_labels, emotion_probs)
                }
                
                # Get dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                
                results.append({
                    "face_id": i + 1,
                    "face_coordinates": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "dominant_emotion": dominant_emotion,
                    "confidence": round(confidence, 2),
                    "all_emotions": {k: round(v, 2) for k, v in emotion_scores.items()}
                })
            
            return {
                "success": True,
                "faces_detected": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {str(e)}")
            return {"error": f"Prediction error: {str(e)}", "success": False}

# Initialize Flask app with SocketIO for real-time communication
app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
emotion_model = None
camera = None
realtime_active = False

def initialize_model(model_path):
    """Initialize the emotion detection model"""
    global emotion_model
    try:
        emotion_model = EmotionDetectionModel(model_path)
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_active = False
        
    def start_camera(self, camera_id=0):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            self.is_active = True
            logger.info(f"Camera {camera_id} started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_active = False
        logger.info("Camera stopped")
    
    def get_frame(self):
        """Get a frame from camera"""
        if not self.is_active or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

camera_manager = CameraManager()

# HTML template for web interface with real-time support
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Emotion Detection Server</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-panel { flex: 1; min-width: 300px; }
        .realtime-panel { flex: 1; min-width: 300px; }
        .results-panel { width: 100%; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-area.dragover { border-color: #4CAF50; background-color: #f0fff0; }
        .result { background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .emotion-bar { background: #ddd; height: 20px; margin: 5px 0; border-radius: 10px; overflow: hidden; }
        .emotion-fill { background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; border-radius: 10px; transition: width 0.3s ease; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; margin: 5px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .stop-btn { background: #f44336; }
        .stop-btn:hover { background: #da190b; }
        .error { color: red; background: #ffebee; padding: 10px; border-radius: 5px; }
        .success { color: green; background: #e8f5e8; padding: 10px; border-radius: 5px; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.active { background: #e8f5e8; color: green; }
        .status.inactive { background: #ffebee; color: red; }
        #videoFeed { max-width: 100%; height: auto; border-radius: 10px; }
        .realtime-controls { text-align: center; margin: 20px 0; }
        .face-box { position: absolute; border: 2px solid #4CAF50; background: rgba(76, 175, 80, 0.1); }
        .face-label { position: absolute; background: #4CAF50; color: white; padding: 2px 8px; font-size: 12px; border-radius: 3px; }
        .video-container { position: relative; display: inline-block; }
        h1 { text-align: center; color: #333; }
        h2 { color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-item { text-align: center; }
        .stat-number { font-size: 24px; font-weight: bold; color: #4CAF50; }
    </style>
</head>
<body>
    <h1>🎭 Real-time Emotion Detection Server</h1>
    
    <div class="container">
        <!-- File Upload Panel -->
        <div class="panel upload-panel">
            <h2>📸 Image Upload</h2>
            <div class="upload-area" id="uploadArea">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <button onclick="document.getElementById('fileInput').click()">Choose Image</button>
                <p>or drag and drop an image here</p>
            </div>
        </div>
        
        <!-- Real-time Camera Panel -->
        <div class="panel realtime-panel">
            <h2>📹 Real-time Detection</h2>
            <div class="status" id="cameraStatus">Camera: Inactive</div>
            
            <div class="realtime-controls">
                <button id="startBtn" onclick="startRealtime()">Start Camera</button>
                <button id="stopBtn" onclick="stopRealtime()" class="stop-btn" disabled>Stop Camera</button>
            </div>
            
            <div class="video-container">
                <canvas id="videoCanvas" width="640" height="480" style="max-width: 100%; border-radius: 10px; background: #000;"></canvas>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number" id="fpsCounter">0</div>
                    <div>FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="faceCounter">0</div>
                    <div>Faces</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Results Panel -->
    <div class="panel results-panel">
        <h2>📊 Detection Results</h2>
        <div id="results"></div>
    </div>

    <script>
        const socket = io();
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const results = document.getElementById('results');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const cameraStatus = document.getElementById('cameraStatus');
        const fpsCounter = document.getElementById('fpsCounter');
        const faceCounter = document.getElementById('faceCounter');
        
        let isRealTimeActive = false;
        let frameCount = 0;
        let lastTime = Date.now();

        // File upload handlers
        fileInput.addEventListener('change', handleFile);
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);

        function handleDragOver(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        }

        function handleFile(event) {
            const file = event.target.files[0];
            if (file) {
                handleFileUpload(file);
            }
        }

        function handleFileUpload(file) {
            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => displayResults(data))
            .catch(error => {
                results.innerHTML = '<div class="error">Error: ' + error + '</div>';
            });
        }

        function startRealtime() {
            socket.emit('start_realtime');
            isRealTimeActive = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            cameraStatus.textContent = 'Camera: Starting...';
            cameraStatus.className = 'status active';
        }

        function stopRealtime() {
            socket.emit('stop_realtime');
            isRealTimeActive = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            cameraStatus.textContent = 'Camera: Inactive';
            cameraStatus.className = 'status inactive';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            fpsCounter.textContent = '0';
            faceCounter.textContent = '0';
        }

        // Socket event handlers
        socket.on('camera_started', function(data) {
            cameraStatus.textContent = 'Camera: Active';
            cameraStatus.className = 'status active';
        });

        socket.on('camera_error', function(data) {
            cameraStatus.textContent = 'Camera: Error - ' + data.error;
            cameraStatus.className = 'status inactive';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        });

        socket.on('realtime_result', function(data) {
            if (data.frame) {
                drawFrame(data.frame, data.results);
                updateStats(data.results);
            }
        });

        function drawFrame(frameData, results) {
            const img = new Image();
            img.onload = function() {
                // Clear canvas and draw frame
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Draw face boxes and emotions
                if (results && results.success && results.results) {
                    const scaleX = canvas.width / img.width;
                    const scaleY = canvas.height / img.height;
                    
                    results.results.forEach(result => {
                        const coord = result.face_coordinates;
                        const x = coord.x * scaleX;
                        const y = coord.y * scaleY;
                        const w = coord.w * scaleX;
                        const h = coord.h * scaleY;
                        
                        // Draw face rectangle
                        ctx.strokeStyle = '#4CAF50';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h);
                        
                        // Draw emotion label
                        const emotion = result.dominant_emotion;
                        const confidence = result.confidence;
                        const emoji = getEmoji(emotion);
                        const label = `${emoji} ${emotion}`;
                        
                        ctx.fillStyle = '#4CAF50';
                        ctx.fillRect(x, y - 25, label.length * 8, 20);
                        ctx.fillStyle = 'white';
                        ctx.font = '12px Arial';
                        ctx.fillText(label, x + 2, y - 10);
                    });
                }
                
                // Update FPS
                frameCount++;
                const now = Date.now();
                if (now - lastTime >= 1000) {
                    fpsCounter.textContent = frameCount;
                    frameCount = 0;
                    lastTime = now;
                }
            };
            img.src = 'data:image/jpeg;base64,' + frameData;
        }

        function updateStats(results) {
            if (results && results.success) {
                faceCounter.textContent = results.faces_detected || 0;
            }
        }

        function displayResults(data) {
            if (!data.success) {
                results.innerHTML = '<div class="error">❌ ' + data.error + '</div>';
                return;
            }

            let html = '<div class="result">';
            html += '<h3>🎯 Detection Results</h3>';
            html += '<p class="success">✅ Face Detected ';

            data.results.forEach((result, index) => {
                html += '<div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">';
                html += '<h4>Face ' + result.face_id + '</h4>';
                html += '<p><strong>Dominant Emotion:</strong> ' + getEmoji(result.dominant_emotion) + ' ' + result.dominant_emotion + ' (' + result.confidence + '%)</p>';
                html += '<h5>All Emotions:</h5>';
                
                Object.entries(result.all_emotions)
                    .sort(([,a], [,b]) => b - a)
                    .forEach(([emotion, score]) => {
                        html += '<div style="margin: 5px 0;">';
                        html += '<span>' + getEmoji(emotion) + ' ' + emotion + ': ' + score + '%</span>';
                        html += '<div class="emotion-bar">';
                        html += '<div class="emotion-fill" style="width: ' + score + '%"></div>';
                        html += '</div>';
                        html += '</div>';
                    });
                html += '</div>';
            });

            html += '</div>';
            results.innerHTML = html;
        }

        function getEmoji(emotion) {
            const emojis = {
                'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
                'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
            };
            return emojis[emotion] || '🎭';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for emotion prediction"""
    try:
        if emotion_model is None:
            return jsonify({"error": "Model not initialized", "success": False}), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "success": False}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected", "success": False}), 400
        
        # Read and decode image
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Predict emotion
        results = emotion_model.predict_emotion(image)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}", "success": False}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = "healthy" if emotion_model is not None else "model not loaded"
    return jsonify({
        "status": status, 
        "model_loaded": emotion_model is not None,
        "camera_active": camera_manager.is_active,
        "realtime_active": realtime_active
    })

# Real-time processing function
def realtime_emotion_detection():
    """Real-time emotion detection loop"""
    global realtime_active
    
    while realtime_active:
        try:
            frame = camera_manager.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Predict emotions
            results = emotion_model.predict_emotion(frame)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit results to all connected clients
            socketio.emit('realtime_result', {
                'frame': frame_base64,
                'results': results,
                'timestamp': time.time()
            })
            
            # Control frame rate (adjust as needed)
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error in realtime detection: {str(e)}")
            socketio.emit('camera_error', {'error': str(e)})
            break

# SocketIO event handlers
@socketio.on('start_realtime')
def handle_start_realtime():
    """Start real-time emotion detection"""
    global realtime_active
    
    if emotion_model is None:
        emit('camera_error', {'error': 'Model not loaded'})
        return
    
    if camera_manager.start_camera():
        realtime_active = True
        emit('camera_started', {'status': 'Camera started successfully'})
        
        # Start real-time processing in a separate thread
        detection_thread = threading.Thread(target=realtime_emotion_detection)
        detection_thread.daemon = True
        detection_thread.start()
    else:
        emit('camera_error', {'error': 'Failed to start camera'})

@socketio.on('stop_realtime')
def handle_stop_realtime():
    """Stop real-time emotion detection"""
    global realtime_active
    realtime_active = False
    camera_manager.stop_camera()
    emit('camera_stopped', {'status': 'Camera stopped'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

def main():
    """Main function to start the server"""
    print("🎭 Real-time Emotion Detection Server")
    print("=" * 40)
    
    # Get model path from command line argument or user input
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = input("Enter the path to your model file (.h5/.keras/.h5.keras): ").strip()
        
        # Remove quotes if user copied path with quotes
        if model_path.startswith('"') and model_path.endswith('"'):
            model_path = model_path[1:-1]
        elif model_path.startswith("'") and model_path.endswith("'"):
            model_path = model_path[1:-1]
    
    # Initialize model
    if not initialize_model(model_path):
        print("❌ Failed to initialize model. Exiting...")
        sys.exit(1)
    
    print("✅ Model loaded successfully!")
    print("🚀 Starting real-time emotion detection server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("📡 API endpoints:")
    print("   - POST /predict (image upload)")
    print("   - GET /health (server status)")
    print("🎥 Real-time features:")
    print("   - Live camera feed")
    print("   - Real-time emotion detection")
    print("   - WebSocket communication")
    print("\nPress Ctrl+C to stop the server")
    
    # Start Flask-SocketIO server
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        global realtime_active
        realtime_active = False
        camera_manager.stop_camera()

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import tensorflow
        import flask
        import flask_socketio
        import PIL
    except ImportError:
        print("Installing required packages...")
        os.system("pip install tensorflow flask flask-socketio pillow opencv-python")
    
    main()