# asl_translator.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image
import mediapipe as mp
from tensorflow.keras.applications import MobileNetV2

class ASLLetterTranslator:
    def __init__(self, img_height=128, img_width=128):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = None
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = None
        
    def _build_inference_model(self, num_classes):
        """Build the model architecture matching training setup"""
        base_model = MobileNetV2(
            weights="imagenet", 
            include_top=False, 
            input_shape=(self.img_height, self.img_width, 3)
        )
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax")
        ])
        return model
    
    def extract_hand_roi(self, frame):

        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            h, w, _ = frame.shape
            
            x_coords = []
            y_coords = []
            
            for landmark in hand_landmarks.landmark:
                x_coords.append(int(landmark.x * w))
                y_coords.append(int(landmark.y * h))

            padding = 30
            x_min = max(0, min(x_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_min = max(0, min(y_coords) - padding)
            y_max = min(h, max(y_coords) + padding)
            
            box_width = x_max - x_min
            box_height = y_max - y_min
            max_side = max(box_width, box_height)
            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            half_side = max_side // 2
            x_min = max(0, center_x - half_side)
            x_max = min(w, center_x + half_side)
            y_min = max(0, center_y - half_side)
            y_max = min(h, center_y + half_side)
            
            hand_roi = frame[y_min:y_max, x_min:x_max]
            
            return hand_roi, (x_min, y_min, x_max, y_max), hand_landmarks
        
        return None, None, None
    
    def predict_letter(self, image_path):
        """Predict ASL letter from image file"""
        if self.model is None:
            raise ValueError("Model not loaded yet!")

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image from {image_path}")

        hand_roi, _, _ = self.extract_hand_roi(frame)
        
        if hand_roi is None:
            return None, 0.0  
        hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)

        lab = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        hand_rgb = cv2.merge([l,a,b])
        hand_rgb = cv2.cvtColor(hand_rgb, cv2.COLOR_LAB2RGB)

        hand_resized = cv2.resize(hand_rgb, (self.img_width, self.img_height))
        hand_normalized = hand_resized / 255.0
        hand_batch = np.expand_dims(hand_normalized, axis=0)
        

        predictions = self.model.predict(hand_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        predicted_letter = self.class_names[predicted_class]
        
        return predicted_letter, confidence
    
    def predict_from_webcam(self, confidence_threshold=0.6):
      
        if self.model is None:
            raise ValueError("Model not loaded yet!")
        
        cap = cv2.VideoCapture(0)
        
        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        
        print("=" * 60)
        print("ASL Translator")
        print("=" * 60)
        print("Show your hand sign in front of the cam")
        print("Background noise filtered")
        print("Press q to quit")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                break
            
            frame = cv2.flip(frame, 1)
            
            hand_roi, bbox, hand_landmarks = self.extract_hand_roi(frame)
            
            if hand_roi is not None and bbox is not None:
                x_min, y_min, x_max, y_max = bbox
 
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                

                if hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                

                try:
                    hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                    hand_resized = cv2.resize(hand_rgb, (self.img_width, self.img_height))
                    hand_normalized = hand_resized / 255.0
                    hand_batch = np.expand_dims(hand_normalized, axis=0)
                    

                    predictions = self.model.predict(hand_batch, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class]
                    predicted_letter = self.class_names[predicted_class]
                    

                    if confidence > confidence_threshold:
                        text = f"Letter: {predicted_letter}"
                        conf_text = f"Confidence: {confidence:.1%}"
                        
                        cv2.putText(frame, text, (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.putText(frame, conf_text, (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                        y_offset = 160
                        cv2.putText(frame, "Top 3:", (50, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        for i, idx in enumerate(top_3_indices):
                            letter = self.class_names[idx]
                            conf = predictions[0][idx]
                            y_offset += 40
                            
                            if i == 0:
                                color = (0, 255, 0)
                            elif i == 1:
                                color = (0, 255, 255)  
                            else:
                                color = (255, 255, 255)  
                            cv2.putText(frame, f"{i+1}. {letter}: {conf:.1%}", (50, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        
                        cv2.putText(frame, "Uncertain", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                        cv2.putText(frame, f"Confidence: {confidence:.1%}", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        cv2.putText(frame, "Hold position steady", (50, 140),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                
                except Exception as e:
                    cv2.putText(frame, f"Processing error", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                
                cv2.putText(frame, "No hand detected", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "Show your hand clearly", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
               
                h, w = frame.shape[:2]
                guide_size = 250
                gx1 = w // 2 - guide_size // 2
                gy1 = h // 2 - guide_size // 2
                gx2 = gx1 + guide_size
                gy2 = gy1 + guide_size
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)
                cv2.putText(frame, "Place hand here", (gx1, gy1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
           
            cv2.imshow('ASL Letter Translator - MediaPipe', frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
  
        cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()
    
    def save_model(self, filepath):
        if self.model is not None:
            self.model.save(filepath)
            classes_path = filepath.replace('.h5', '_classes.npy').replace('.keras', '_classes.npy')
            np.save(classes_path, self.class_names)
            print(f"Model saved to {filepath}")
            print(f"Classes saved to {classes_path}")
    
    def load_model(self, filepath):
        base_dir = os.path.dirname(filepath)
        classes_filename = os.path.basename(filepath).replace('.h5', '_classes.npy').replace('.keras', '_classes.npy')

        possible_paths = [
            os.path.join(base_dir, classes_filename), 
            classes_filename,  
            "asl_model_classes.npy"  
        ]
        
        classes_path = None
        for path in possible_paths:
            if os.path.exists(path):
                classes_path = path
                break
        
        if classes_path:
            self.class_names = np.load(classes_path, allow_pickle=True)
            self.label_encoder.fit(self.class_names)
            num_classes = len(self.class_names)
            print(f"Loaded {num_classes} classes from {classes_path}")
            print(f"Classes: {list(self.class_names)}")
        else:
            print(f"Warning: Class names file not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            num_classes = 29
            self.class_names = None
        

        self.model = self._build_inference_model(num_classes)
        
        self.model.load_weights(filepath)
        
        print(f"Model loaded from {filepath}")
        print("MediaPipe Hands ready")
    
    def __del__(self):
        if self.hands:
            self.hands.close()



if __name__ == "__main__":
    print("ASL Letter Translator")
    print("=" * 50)
    print("\n To use:")
    print("  translator = ASLLetterTranslator()")
    print("  translator.load_model('models/asl_model_final.keras')")
    print("  translator.predict_from_webcam()")
    print("=" * 50)