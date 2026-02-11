# Rewritten version to use a local .h5 or .keras model for prediction

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class H5ModelValidator:
    def __init__(self, validation_folder, model_path):
        self.validation_folder = validation_folder
        self.model_path = model_path
        self.model = load_model(model_path)
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.results = []
        self.predictions = []
        self.true_labels = []
        self.image_paths = []
        self.processing_times = []

        self.output_dir = "validation_reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_image(self, image_path, target_size=(48, 48)):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict_single_image(self, image_path):
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                raise ValueError("Invalid image")

            start_time = time.time()
            preds = self.model.predict(image)[0]
            processing_time = time.time() - start_time

            predicted_emotion = self.emotion_classes[np.argmax(preds)]
            confidence = np.max(preds)

            return {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'all_emotions': dict(zip(self.emotion_classes, preds)),
                'processing_time': processing_time,
                'success': True
            }

        except Exception as e:
            return {
                'predicted_emotion': None,
                'confidence': 0.0,
                'all_emotions': {},
                'processing_time': 0.0,
                'success': False,
                'error': str(e)
            }

    def load_dataset(self):
        dataset = {}
        for class_name in os.listdir(self.validation_folder):
            class_path = os.path.join(self.validation_folder, class_name)
            if not os.path.isdir(class_path):
                continue
            normalized_class = class_name.lower()
            if normalized_class not in self.emotion_classes:
                continue
            dataset[normalized_class] = []
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    dataset[normalized_class].append(os.path.join(class_path, file_name))
        return dataset

    def validate_dataset(self, dataset):
        for true_class, image_paths in dataset.items():
            for image_path in image_paths:
                prediction = self.predict_single_image(image_path)
                result_record = {
                    'image_path': image_path,
                    'true_label': true_class,
                    'predicted_label': prediction['predicted_emotion'],
                    'confidence': prediction['confidence'],
                    'processing_time': prediction['processing_time'],
                    'success': prediction['success']
                }
                for emotion, score in prediction.get('all_emotions', {}).items():
                    result_record[f'{emotion}_score'] = score

                self.results.append(result_record)
                if prediction['success']:
                    self.true_labels.append(true_class)
                    self.predictions.append(prediction['predicted_emotion'])
                    self.image_paths.append(image_path)
                    self.processing_times.append(prediction['processing_time'])

    def run_validation(self):
        dataset = self.load_dataset()
        if not dataset:
            print("No valid dataset found.")
            return

        self.validate_dataset(dataset)

        print("Validation complete.")
        print(f"Accuracy: {accuracy_score(self.true_labels, self.predictions):.2f}")
        print(classification_report(self.true_labels, self.predictions, target_names=self.emotion_classes))


def main():
    validation_folder = input("Enter path to validation folder: ").strip()
    model_path = input("Enter path to .h5 or .keras model file: ").strip()
    if not os.path.exists(validation_folder):
        print("Invalid validation folder.")
        return
    if not os.path.exists(model_path):
        print("Invalid model file.")
        return

    validator = H5ModelValidator(validation_folder, model_path)
    validator.run_validation()

if __name__ == '__main__':
    main()