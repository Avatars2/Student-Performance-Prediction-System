from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Grade mapping for better display
grade_info = {
    'A': {'description': 'Excellent', 'color': '#28a745', 'range': '90-100%'},
    'B': {'description': 'Good', 'color': '#17a2b8', 'range': '80-89%'},
    'C': {'description': 'Average', 'color': '#ffc107', 'range': '70-79%'},
    'D': {'description': 'Below Average', 'color': '#fd7e14', 'range': '60-69%'},
    'F': {'description': 'Fail', 'color': '#dc3545', 'range': 'Below 60%'}
}

def validate_input(study_hours, attendance, previous_marks, assignments_completed, participation):
    """Validate input values"""
    errors = []
    
    if not study_hours or float(study_hours) < 0 or float(study_hours) > 24:
        errors.append("Study hours must be between 0 and 24")
    
    if not attendance or float(attendance) < 0 or float(attendance) > 100:
        errors.append("Attendance must be between 0 and 100")
    
    if not previous_marks or float(previous_marks) < 0 or float(previous_marks) > 100:
        errors.append("Previous marks must be between 0 and 100")
    
    if not assignments_completed or float(assignments_completed) < 0 or float(assignments_completed) > 10:
        errors.append("Assignments completed must be between 0 and 10")
    
    if not participation or float(participation) < 0 or float(participation) > 10:
        errors.append("Participation score must be between 0 and 10")
    
    return errors

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model is None or scaler is None:
        return render_template('result.html', 
                             prediction=None, 
                             error="Model not loaded. Please train the model first.")
    
    try:
        # Get form data
        study_hours = request.form.get('study_hours')
        attendance = request.form.get('attendance')
        previous_marks = request.form.get('previous_marks')
        assignments_completed = request.form.get('assignments_completed')
        participation = request.form.get('participation')
        
        # Validate input
        errors = validate_input(study_hours, attendance, previous_marks, assignments_completed, participation)
        
        if errors:
            return render_template('result.html', 
                                 prediction=None, 
                                 error="Invalid input: " + ", ".join(errors))
        
        # Convert to float and create feature array
        features = np.array([[
            float(study_hours),
            float(attendance),
            float(previous_marks),
            float(assignments_completed),
            float(participation)
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(prediction_proba) * 100
        
        # Get grade information
        grade_data = grade_info.get(prediction, {'description': 'Unknown', 'color': '#6c757d', 'range': 'N/A'})
        
        # Prepare result data
        result = {
            'grade': prediction,
            'description': grade_data['description'],
            'color': grade_data['color'],
            'range': grade_data['range'],
            'confidence': round(confidence, 2),
            'input_data': {
                'study_hours': float(study_hours),
                'attendance': float(attendance),
                'previous_marks': float(previous_marks),
                'assignments_completed': float(assignments_completed),
                'participation': float(participation)
            }
        }
        
        return render_template('result.html', prediction=result, error=None)
        
    except Exception as e:
        return render_template('result.html', 
                             prediction=None, 
                             error=f"An error occurred during prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['study_hours', 'attendance', 'previous_marks', 'assignments_completed', 'participation']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create feature array
        features = np.array([[
            float(data['study_hours']),
            float(data['attendance']),
            float(data['previous_marks']),
            float(data['assignments_completed']),
            float(data['participation'])
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(prediction_proba) * 100
        
        return jsonify({
            'grade': prediction,
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Return model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        feature_names = ['study_hours', 'attendance', 'previous_marks', 'assignments_completed', 'participation']
        feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
        
        return jsonify({
            'model_type': 'Random Forest Classifier',
            'features': feature_names,
            'feature_importance': feature_importance,
            'classes': model.classes_.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
