# use_asl_translator.py
from asl_translator import ASLLetterTranslator
import numpy as np
import tensorflow as tf

def main():
    translator = ASLLetterTranslator()
    translator.load_model("models/asl_model_final.keras") 
    translator.predict_from_webcam()

if __name__ == "__main__":
    main()
