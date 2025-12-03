This American Sign Language translator's model was trained on the ASL Alphabet dataset from Kaggle.
Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Features:
- Hand gesture recognition through MediaPipe Hands
- Convolutional Neural Network trained with TensorFlow and transfer learning thorough MobileNetV2
- Webcam-based real-time prediction

Setup:
py -m pip install -r requirements.txt

  To train the model:
  py -m pip install -r requirements.txt
  py train_asl_model.py
  py save_asl_classes.py

  To run the translator:
  py use_asl_translator.py
  
