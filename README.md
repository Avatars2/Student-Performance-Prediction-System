# Student Performance Prediction System

A complete end-to-end Machine Learning web application that predicts student academic performance using Flask and Random Forest Classifier.

## 🎯 Project Overview

This system uses Machine Learning to predict student grades based on various academic factors including study hours, attendance, previous marks, assignments completed, and class participation. The application provides a user-friendly web interface for inputting student data and receiving instant predictions with confidence scores.

## 📊 Features

- **Machine Learning Model**: Random Forest Classifier with high accuracy
- **Web Interface**: Clean, responsive UI built with Flask, HTML5, and CSS3
- **Input Validation**: Comprehensive validation for all input fields
- **Prediction Results**: Detailed predictions with confidence scores and personalized recommendations
- **API Endpoints**: RESTful API for programmatic access
- **Responsive Design**: Mobile-friendly interface
- **Print Support**: Printable result pages

## 🛠️ Technology Stack

- **Backend**: Python 3.7+, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript
- **Model**: Random Forest Classifier
- **Data Processing**: StandardScaler for feature normalization

## 📁 Project Structure

```
student-performance/
├── app.py                 # Flask application main file
├── train_model.py         # Model training script
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── dataset.csv           # Student dataset
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── templates/
│   ├── index.html       # Input form page
│   └── result.html      # Prediction results page
└── static/
    └── style.css        # Styling for the application
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or download the project** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd student-performance
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the machine learning model**:
   ```bash
   python train_model.py
   ```
   
   This will:
   - Load and preprocess the dataset
   - Train the Random Forest model
   - Display accuracy metrics and feature importance
   - Save the trained model as `model.pkl`
   - Save the scaler as `scaler.pkl`

6. **Run the Flask application**:
   ```bash
   python app.py
   ```

7. **Access the application**:
   Open your web browser and go to: `http://127.0.0.1:5000`

## 📱 How to Use

1. **Open the application** in your web browser
2. **Fill in the student information**:
   - Study Hours (per day): 0-24 hours
   - Attendance: 0-100%
   - Previous Marks: 0-100%
   - Assignments Completed: 0-10
   - Participation Score: 0-10
3. **Click "Predict Performance"** to get the prediction
4. **View the results** including:
   - Predicted grade (A, B, C, D, or F)
   - Confidence percentage
   - Personalized recommendations
   - Input summary

## 🎓 Grade System

- **A**: Excellent (90-100%)
- **B**: Good (80-89%)
- **C**: Average (70-79%)
- **D**: Below Average (60-69%)
- **F**: Fail (Below 60%)

## 🔧 API Endpoints

### POST /api/predict
Make predictions programmatically.

**Request Body:**
```json
{
    "study_hours": 8.5,
    "attendance": 95,
    "previous_marks": 78,
    "assignments_completed": 10,
    "participation": 8
}
```

**Response:**
```json
{
    "grade": "A",
    "confidence": 92.5
}
```

### GET /model_info
Get information about the trained model.

**Response:**
```json
{
    "model_type": "Random Forest Classifier",
    "features": ["study_hours", "attendance", "previous_marks", "assignments_completed", "participation"],
    "feature_importance": {...},
    "classes": ["A", "B", "C", "D"]
}
```

## 📈 Model Performance

The Random Forest Classifier typically achieves:
- **Accuracy**: ~95% on test data
- **Features Used**: 5 academic performance indicators
- **Cross-validation**: 5-fold stratified validation
- **Feature Importance**: Previous marks and study hours are typically the most important features

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Make sure you've run `python train_model.py` first
   - Check that `model.pkl` and `scaler.pkl` files exist in the project directory

2. **Port already in use**:
   - Change the port in `app.py`:
     ```python
     app.run(debug=True, host='127.0.0.1', port=5001)
     ```

3. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.7+)

4. **Permission errors**:
   - Run the application with appropriate permissions
   - On Windows, run Command Prompt as Administrator

### Debug Mode

The application runs in debug mode by default. You'll see detailed error messages in the terminal if something goes wrong.

## 🔄 Retraining the Model

To retrain the model with new data:

1. **Update the dataset** in `dataset.csv`
2. **Run the training script**:
   ```bash
   python train_model.py
   ```
3. **Restart the Flask application**:
   ```bash
   python app.py
   ```

## 🎨 Customization

### Adding New Features

1. **Update the dataset** with new columns
2. **Modify `train_model.py`** to include new features
3. **Update the HTML forms** in `templates/index.html`
4. **Modify the Flask app** in `app.py` to handle new inputs

### Changing the Model

Replace the Random Forest Classifier in `train_model.py` with any scikit-learn classifier:

```python
from sklearn.linear_model import LogisticRegression
# or
from sklearn.svm import SVC
# or
from sklearn.ensemble import GradientBoostingClassifier
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify the model has been trained before running the app
4. Check the terminal output for detailed error messages

## 🔄 Version History

- **v1.0.0**: Initial release with Random Forest model and Flask web interface
- Features: Grade prediction, confidence scoring, responsive design, API endpoints

---

**Happy Predicting! 🎓✨**
# Student-Performance-Prediction-System
