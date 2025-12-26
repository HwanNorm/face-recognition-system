import cv2
import os
import pandas as pd
from deepface import DeepFace
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DeepFaceRecognitionSystem:
    def __init__(self):
        self.setup_directories()
        # Available models: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
        self.model_name = "Facenet512"  # Best accuracy
        self.detector_backend = "opencv"  # opencv, ssd, dlib, mtcnn, retinaface
        self.database_path = "dataset"
        print(f"✓ Using model: {self.model_name}")
        print(f"✓ Using detector: {self.detector_backend}")

    def setup_directories(self):
        """Create necessary directories"""
        directories = ["dataset", "models"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✓ Project directories created!")

    def capture_images_for_person(self, person_name, num_images=10):
        """Capture images for a specific person"""
        print(f"\n📸 Starting image capture for: {person_name}")
        print("Instructions:")
        print("- Look at the camera")
        print("- Press SPACE to take a photo")
        print("- Press Q to quit early")
        print("- Try different angles/expressions for better recognition")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return False
        
        # Create directory for person
        person_dir = f"dataset/{person_name}"
        os.makedirs(person_dir, exist_ok=True)
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read from camera")
                break
            
            # Show instructions on frame
            text = f"Capturing for {person_name} - Photo {count+1}/{num_images} - Press SPACE"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capture Images', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key to capture
                img_path = f"{person_dir}/{person_name}_{count+1}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"✓ Captured: {img_path}")
                count += 1
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✓ Captured {count} images for {person_name}")
        return count > 0

    def verify_dataset(self):
        """Verify that dataset is ready for DeepFace"""
        print("\n🔍 Verifying dataset...")
        
        if not os.path.exists(self.database_path):
            print("❌ No dataset directory found")
            return False
        
        people_count = 0
        total_images = 0
        
        for person_name in os.listdir(self.database_path):
            person_path = os.path.join(self.database_path, person_name)
            if not os.path.isdir(person_path):
                continue
            
            images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(images)
            
            if image_count > 0:
                people_count += 1
                total_images += image_count
                print(f"  ✓ {person_name}: {image_count} images")
            else:
                print(f"  ⚠️ {person_name}: No images found")
        
        print(f"\n✓ Dataset verified: {people_count} people, {total_images} total images")
        return people_count > 0

    def build_database(self):
        """Build face database for DeepFace"""
        print(f"\n🧠 Building face database with {self.model_name}...")
        
        try:
            # DeepFace will automatically build representations
            # This happens when we first call find() or verify()
            print("✓ Database will be built on first recognition")
            return True
        except Exception as e:
            print(f"❌ Error building database: {e}")
            return False

    def recognize_face_in_image(self, image_path):
        """Recognize face in a single image"""
        try:
            # Find similar faces in database
            result = DeepFace.find(
                img_path=image_path,
                db_path=self.database_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            if len(result) > 0 and len(result[0]) > 0:
                # Get the best match
                best_match = result[0].iloc[0]
                identity = best_match['identity']
                distance = best_match['distance']
                
                # Extract person name from path
                person_name = os.path.basename(os.path.dirname(identity))
                confidence = max(0, 100 - (distance * 100))
                
                return person_name, confidence, distance
            else:
                return "Unknown", 0, 1.0
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0, 1.0

    def run_realtime_recognition(self):
        """Run real-time face recognition"""
        print(f"\n🎥 Starting real-time recognition with {self.model_name}...")
        print("Press 'q' to quit")
        
        # Verify dataset first
        if not self.verify_dataset():
            print("❌ Please add some people to the dataset first")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        frame_count = 0
        recognition_interval = 30  # Recognize every 30 frames for performance
        last_recognition = {"name": "Unknown", "confidence": 0}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform recognition every N frames
            if frame_count % recognition_interval == 0:
                # Save current frame temporarily
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Recognize face
                name, confidence, distance = self.recognize_face_in_image(temp_path)
                last_recognition = {"name": name, "confidence": confidence, "distance": distance}
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Debug info
                print(f"DEBUG: {name} - Confidence: {confidence:.1f}% - Distance: {distance:.3f}")
            
            # Draw recognition results
            frame = self.draw_recognition_results(frame, last_recognition)
            
            # Show frame
            cv2.imshow('DeepFace Real-Time Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

    def draw_recognition_results(self, frame, recognition_result):
        """Draw recognition results on frame"""
        name = recognition_result["name"]
        confidence = recognition_result["confidence"]
        
        # Choose color based on recognition
        if name == "Unknown":
            color = (0, 0, 255)  # Red
            label = "Unknown"
        else:
            color = (0, 255, 0)  # Green
            label = f"{name} ({confidence:.1f}%)"
        
        # Draw label at top of frame
        cv2.rectangle(frame, (10, 10), (400, 60), color, cv2.FILLED)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def test_with_image(self, image_path):
        """Test recognition with a specific image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"\n🔍 Testing recognition with: {image_path}")
        
        name, confidence, distance = self.recognize_face_in_image(image_path)
        
        print(f"Result: {name}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Distance: {distance:.3f}")
        
        # Show the image with result
        img = cv2.imread(image_path)
        if img is not None:
            label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else "Unknown"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(img, (10, 10), (400, 60), color, cv2.FILLED)
            cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Recognition Test', img)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def change_model(self):
        """Change the recognition model"""
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            marker = " ← CURRENT" if model == self.model_name else ""
            print(f"{i}. {model}{marker}")
        
        try:
            choice = int(input(f"\nChoose model (1-{len(models)}): ")) - 1
            if 0 <= choice < len(models):
                self.model_name = models[choice]
                print(f"✓ Model changed to: {self.model_name}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")

    def show_menu(self):
        """Interactive menu"""
        while True:
            print("\n" + "="*60)
            print("🚀 DEEPFACE RECOGNITION SYSTEM")
            print("="*60)
            print("1. 📸 Capture images for new person")
            print("2. 🔍 Verify dataset")
            print("3. 🎥 Start real-time recognition")
            print("4. 🖼️  Test with image file")
            print("5. ⚙️  Change recognition model")
            print("6. ❌ Exit")
            print("="*60)
            print(f"Current model: {self.model_name}")
            print("="*60)
            
            choice = input("Choose an option (1-6): ").strip()
            
            if choice == '1':
                person_name = input("Enter person's name: ").strip()
                if person_name:
                    num_images = 10
                    try:
                        num_images = int(input(f"Number of images to capture (default 10): ") or "10")
                    except ValueError:
                        num_images = 10
                    
                    self.capture_images_for_person(person_name, num_images)
                else:
                    print("❌ Please enter a valid name")
            
            elif choice == '2':
                self.verify_dataset()
            
            elif choice == '3':
                self.run_realtime_recognition()
            
            elif choice == '4':
                image_path = input("Enter path to image file: ").strip()
                self.test_with_image(image_path)
            
            elif choice == '5':
                self.change_model()
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    """Main function"""
    print("🚀 Welcome to DeepFace Recognition System!")
    print("This uses state-of-the-art deep learning models!")
    
    # Check required libraries
    try:
        import cv2
        import pandas as pd
        from deepface import DeepFace
        print("✓ All required libraries are installed")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("Please install required libraries:")
        print("pip install deepface opencv-python pandas tensorflow")
        return
    
    # Create and run system
    face_system = DeepFaceRecognitionSystem()
    face_system.show_menu()

if __name__ == "__main__":
    main()