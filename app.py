from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Initialize ML components
vectorizer = None
models = {}
model_performance = {}

def preprocess_text(text):
    """Preprocess the text by removing punctuation, digits, and converting to lowercase"""
    if not text:
        return ""
    # Remove punctuation and digits
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    return text

def create_sample_data():
    """Create comprehensive sample training data for spam detection"""
    # Expanded sample spam emails
    spam_emails = [
        "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
        "URGENT: Your account will be suspended. Verify your information immediately!",
        "Make $5000 per week working from home! No experience needed!",
        "Free iPhone! Click here to get your free iPhone today only!",
        "You are the winner of our lottery! Claim your million dollars now!",
        "Special promotion: Buy now and get 90% discount! Limited time offer!",
        "Your credit score needs improvement. Click here for instant approval!",
        "Make money fast! No investment required! Work from home opportunity!",
        "You have been selected for our exclusive offer! Don't miss out!",
        "Get rich quick with this amazing opportunity! Click now to learn more!",
        "Free trial! Limited time offer! Sign up now for exclusive benefits!",
        "You've been chosen! Claim your free gift card worth $500 today!",
        "Emergency: Your account has been compromised. Verify immediately!",
        "Earn cash from home! No experience needed! Start earning today!",
        "Exclusive deal just for you! 50% off all products! Shop now!",
        "Lose weight fast with this miracle pill! Results guaranteed!",
        "Your package delivery failed. Click to reschedule and avoid fees!",
        "Bank alert: Unusual login detected. Secure your account now!",
        "Investment opportunity: Double your money in 30 days!",
        "You qualify for a government grant! Apply now for free money!",
        "Limited time offer! Buy one get one free! Shop now!",
        "Your account will be deleted. Confirm your details to keep it active!",
        "Make money online with no investment required! Start today!",
        "You have won a luxury car! Claim your prize immediately!",
        "Special discount for valued customers! Limited stock available!"
    ]
    
    # Expanded sample safe (ham) emails
    safe_emails = [
        "Hi John, just checking in about our meeting tomorrow at 2 PM.",
        "Your order #12345 has been shipped. Tracking number: 123456789.",
        "Meeting reminder: Team sync tomorrow at 10 AM in conference room B.",
        "Thanks for your email. I'll get back to you with the information.",
        "The project deadline has been extended to next Friday.",
        "Please find attached the report you requested last week.",
        "Lunch tomorrow at 12:30? Let me know if that works for you.",
        "Your appointment has been confirmed for next Monday at 3 PM.",
        "I wanted to follow up on our conversation from yesterday.",
        "The documents have been reviewed and approved by the team.",
        "Can we schedule a call to discuss the project requirements?",
        "Thanks for your help with the presentation yesterday.",
        "The software update has been completed successfully.",
        "Please review the attached proposal and let me know your thoughts.",
        "Looking forward to our collaboration on the new initiative.",
        "Hi team, here are the minutes from today's meeting.",
        "Your subscription will renew automatically next month.",
        "Welcome to our service! Here's how to get started.",
        "The files you requested are ready for download.",
        "Reminder: Your payment is due next week.",
        "The conference call has been scheduled for tomorrow at 3 PM.",
        "Please find the updated project timeline attached.",
        "Thanks for your feedback on the design mockups.",
        "The budget proposal has been approved by management.",
        "Let me know if you need any additional information."
    ]
    
    # Create DataFrame
    emails = spam_emails + safe_emails
    labels = [1] * len(spam_emails) + [0] * len(safe_emails)  # 1 for spam, 0 for safe
    
    df = pd.DataFrame({
        'email': emails,
        'label': labels
    })
    
    return df

def train_models():
    """Train all three ML models"""
    global vectorizer, models, model_performance
    
    # Create sample data
    df = create_sample_data()
    
    # Preprocess emails
    df['cleaned_email'] = df['email'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_email'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Initialize models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'naive_bayes': MultinomialNB(),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }
    
    # Train and evaluate each model
    model_performance = {}
    
    for model_name, model in models.items():
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        model_performance[model_name] = {
            'accuracy': accuracy,
            'model': model
        }
        
        print(f"{model_name.replace('_', ' ').title()} trained with accuracy: {accuracy:.4f}")
    
    # Save models and vectorizer
    joblib.dump(models, 'spam_models.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(model_performance, 'model_performance.pkl')

def load_models():
    """Load the trained models and vectorizer"""
    global vectorizer, models, model_performance
    
    # Check if model files exist
    models_exist = os.path.exists('spam_models.pkl')
    vectorizer_exists = os.path.exists('vectorizer.pkl')
    performance_exists = os.path.exists('model_performance.pkl')
    
    if models_exist and vectorizer_exists and performance_exists:
        models = joblib.load('spam_models.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        model_performance = joblib.load('model_performance.pkl')
        print("Models loaded successfully")
        
        # Print model performances
        for model_name, perf in model_performance.items():
            print(f"{model_name.replace('_', ' ').title()} accuracy: {perf['accuracy']:.4f}")
    else:
        print("Training new models...")
        train_models()

def predict_email_all_models(email_text):
    """Predict if email is spam or safe using all three models working together"""
    global vectorizer, models
    
    if vectorizer is None or not models:
        load_models()
    
    # Preprocess the input text
    cleaned_text = preprocess_text(email_text)
    
    # Transform using TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Get predictions from all models
    predictions = {}
    probabilities = {}
    confidence_scores = {}
    
    for model_name, model in models.items():
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        predictions[model_name] = prediction
        probabilities[model_name] = probability
        confidence_scores[model_name] = probability[1] if prediction == 1 else probability[0]
    
    # Calculate combined confidence (average of all model confidences)
    combined_confidence = np.mean(list(confidence_scores.values()))
    
    # Use weighted voting based on model accuracy
    weighted_spam_score = 0
    total_weight = 0
    
    for model_name, prediction in predictions.items():
        model_acc = model_performance[model_name]['accuracy']
        weighted_spam_score += prediction * model_acc
        total_weight += model_acc
    
    # Normalize the weighted score
    normalized_score = weighted_spam_score / total_weight
    
    # Final decision based on weighted voting
    final_prediction = 1 if normalized_score > 0.5 else 0
    result = 'spam' if final_prediction == 1 else 'safe'
    
    return {
        'final_result': result,
        'final_confidence': combined_confidence,
        'model_predictions': predictions,
        'model_probabilities': probabilities,
        'model_confidence_scores': confidence_scores,
        'model_performance': model_performance,
        'weighted_score': normalized_score
    }

# Your existing routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/aboutus')
def about():
    return render_template('aboutus.html')

@app.route('/contactus')
def contact():
    return render_template('contactus.html')

@app.route('/models_info')
def models_info():
    """Page showing information about all models"""
    return render_template('models_info.html', model_performance=model_performance)

# Detection route using all models jointly
@app.route('/detect', methods=['POST'])
def detect():
    email = request.form.get('email')
    email_body = request.form.get('email_body')
    
    # Check if email body is provided
    if not email_body:
        return render_template('detection.html', 
                             result='safe', 
                             email=email, 
                             email_body=email_body,
                             confidence=0.0)
    
    # Check if models are loaded
    if vectorizer is None or not models:
        load_models()
    
    # Make prediction using all models jointly
    prediction_result = predict_email_all_models(email_body)
    result = prediction_result['final_result']
    confidence = prediction_result['final_confidence']
    model_predictions = prediction_result['model_predictions']
    model_probabilities = prediction_result['model_probabilities']
    model_confidence_scores = prediction_result['model_confidence_scores']
    weighted_score = prediction_result['weighted_score']
    
    print(f"Prediction: {result} (confidence: {confidence:.4f}, weighted_score: {weighted_score:.4f})")
    
    return render_template('detection.html', 
                         result=result, 
                         email=email, 
                         email_body=email_body,
                         confidence=confidence,
                         model_predictions=model_predictions,
                         model_probabilities=model_probabilities,
                         model_confidence_scores=model_confidence_scores,
                         model_performance=model_performance,
                         weighted_score=weighted_score)

if __name__ == '__main__':
    # Load models when starting the app
    load_models()
    app.run(debug=True, port=5000)