import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import librosa
import os
from imblearn.over_sampling import SMOTE

# Expert System for Parkinson's Disease Prediction
class ParkinsonExpertSystem:
    def __init__(self, model_path=None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.rules = {
            'tremor': 0.41,  # Weight based on prevalence in PD (41% vs <1% in controls)
            'constipation': 0.37,  # 37% vs 23% in controls
            'depression': 0.18,  # 18% vs 10% in controls
        }
        self.class_names = ['Healthy', 'Parkinson']
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load_model()

    def extract_mfcc(self, audio_path, max_length=100, n_mfcc=13):
        """Extract MFCC features from audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            if mfcc.shape[1] < max_length:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :max_length]
            return mfcc.flatten()
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def preprocess_data(self, voice_features, clinical_data):
        """Combine voice and clinical features, scale them"""
        clinical_features = np.array([self.rules.get(symptom, 0) for symptom in clinical_data])
        features = np.concatenate([voice_features, clinical_features])
        features = self.scaler.fit_transform(features.reshape(1, -1))
        return features

    def rule_based_inference(self, clinical_data):
        """Apply rule-based scoring for initial assessment"""
        score = sum(self.rules.get(symptom, 0) for symptom in clinical_data)
        return 'High Risk' if score > 0.5 else 'Low Risk'

    def train(self, voice_files, clinical_data, labels, save_path='parkinson_model.pkl'):
        """Train the model with voice and clinical data"""
        X = []
        for voice_file, clinical in zip(voice_files, clinical_data):
            mfcc = self.extract_mfcc(voice_file)
            if mfcc is not None:
                features = self.preprocess_data(mfcc, clinical)
                X.append(features.flatten())
        
        X = np.array(X)
        y = np.array(labels)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Training Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Save model
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)

    def predict(self, voice_file, clinical_data):
        """Predict PD risk for a single sample"""
        mfcc = self.extract_mfcc(voice_file)
        if mfcc is None:
            return None, None
        
        features = self.preprocess_data(mfcc, clinical_data)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        rule_result = self.rule_based_inference(clinical_data)
        return {
            'ML_Prediction': self.class_names[prediction],
            'PD_Probability': probability[1],
            'Rule_Based_Risk': rule_result
        }

    def load_model(self):
        """Load pre-trained model"""
        import pickle
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']

# Example usage
if __name__ == "__main__":
    # Simulated dataset (replace with actual data)
    voice_files = ['sample1.wav', 'sample2.wav']  # Placeholder audio paths
    clinical_data = [
        ['tremor', 'constipation'],
        ['depression']
    ]
    labels = [1, 0]  # 1: PD, 0: Healthy
    
    # Initialize and train expert system
    expert_system = ParkinsonExpertSystem()
    expert_system.train(voice_files, clinical_data, labels)
    
    # Predict for a new sample
    new_voice = 'new_sample.wav'
    new_clinical = ['tremor', 'depression']
    result = expert_system.predict(new_voice, new_clinical)
    
    if result:
        print("\nPrediction Result:")
        for key, value in result.items():
            print(f"{key}: {value}")