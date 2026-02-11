import os
import sys
from deepface import DeepFace
import cv2

def detect_emotion(image_path):
    """
    Detect emotion in a single image
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Emotion detection results
    """
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Analyze the image using DeepFace
        print(f"Analyzing image: {image_path}")
        print("Processing...")
        
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Extract emotion data
        if isinstance(result, list):
            emotions = result[0]['emotion']
        else:
            emotions = result['emotion']
        
        # Get the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": round(confidence, 2),
            "all_emotions": {k: round(v, 2) for k, v in emotions.items()},
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}", "success": False}

def display_results(results):
    """
    Display emotion detection results in a formatted way
    
    Args:
        results (dict): Results from emotion detection
    """
    if not results.get("success", False):
        print(f"❌ {results.get('error', 'Unknown error')}")
        return
    
    print("\n" + "="*50)
    print("🎭 EMOTION DETECTION RESULTS")
    print("="*50)
    
    # Display dominant emotion
    emotion = results["dominant_emotion"].upper()
    confidence = results["confidence"]
    
    # Emoji mapping for emotions
    emotion_emojis = {
        'ANGRY': '😠',
        'DISGUST': '🤢',
        'FEAR': '😨',
        'HAPPY': '😊',
        'NEUTRAL': '😐',
        'SAD': '😢',
        'SURPRISE': '😲'
    }
    
    emoji = emotion_emojis.get(emotion, '🎭')
    
    print(f"\n🎯 DOMINANT EMOTION: {emoji} {emotion}")
    print(f"📊 CONFIDENCE: {confidence}%")
    
    print(f"\n📋 ALL EMOTION SCORES:")
    print("-" * 30)
    
    # Sort emotions by score (highest first)
    sorted_emotions = sorted(results["all_emotions"].items(), 
                           key=lambda x: x[1], reverse=True)
    
    for emotion, score in sorted_emotions:
        emoji = emotion_emojis.get(emotion.upper(), '🎭')
        # Create a simple progress bar
        bar_length = int(score / 5)  # Scale to 20 chars max
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{emoji} {emotion.capitalize():8} │ {bar} │ {score:5.1f}%")
    
    print("="*50)

def main():
    print("🎭 Simple Emotion Detection Script")
    print("=" * 40)
    
    # Get image path from command line argument or user input
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to your image: ").strip()
        
        # Remove quotes if user copied path with quotes
        if image_path.startswith('"') and image_path.endswith('"'):
            image_path = image_path[1:-1]
        elif image_path.startswith("'") and image_path.endswith("'"):
            image_path = image_path[1:-1]
    
    # Detect emotion
    results = detect_emotion(image_path)
    
    # Display results
    display_results(results)
    
    # Ask if user wants to see the image
    if results.get("success", False):
        show_image = input("\nWould you like to display the image? (y/n): ").strip().lower()
        if show_image in ['y', 'yes']:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    # Add emotion text to image
                    emotion = results["dominant_emotion"].upper()
                    confidence = results["confidence"]
                    
                    # Resize image if too large
                    height, width = img.shape[:2]
                    if height > 800 or width > 800:
                        scale = min(800/height, 800/width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    
                    # Add text overlay
                    cv2.putText(img, f"{emotion}: {confidence}%", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    
                    cv2.imshow('Emotion Detection Result', img)
                    print("Press any key to close the image window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Could not load image for display")
            except Exception as e:
                print(f"Error displaying image: {e}")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import deepface
    except ImportError:
        print("Installing required packages...")
        os.system("pip install deepface opencv-python")
        import deepface
    
    main()