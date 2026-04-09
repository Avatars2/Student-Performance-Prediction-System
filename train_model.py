import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the student dataset"""
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())
    
    # Feature selection - using all available features
    features = ['study_hours', 'attendance', 'previous_marks', 'assignments_completed', 'participation']
    target = 'grade'
    
    X = df[features]
    y = df[target]
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {features}")
    print(f"Target: {target}")
    print(f"Unique grades: {sorted(y.unique())}")
    
    return X, y

def train_model(X, y):
    """Train Random Forest Classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return rf_model, scaler, accuracy

def save_model(model, scaler):
    """Save the trained model and scaler"""
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel and scaler saved successfully!")

def main():
    """Main function to train the model"""
    print("=== Student Performance Prediction Model Training ===\n")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train the model
    model, scaler, accuracy = train_model(X, y)
    
    # Save the model
    save_model(model, scaler)
    
    print(f"\n=== Training Complete ===")
    print(f"Final Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Files saved: model.pkl, scaler.pkl")

if __name__ == "__main__":
    main()
