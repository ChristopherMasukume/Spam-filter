import os
import time
import joblib
from flask import Flask, render_template, request
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge, generate_latest
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Import your classifier

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(lowercase=True)),  # lowercase here
    ('classifier', LogisticRegression())
])
# Initialize Flask
app = Flask(__name__, template_folder='templates')

# ======================
# 1. PROMETHEUS METRICS SETUP
# ======================

PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
PREDICTION_COUNTER = Counter(
    'spam_predictions_total', 
    'Total spam classifications', 
    ['result']  # Added label dimension
)
PREDICTION_TIME = Histogram(
    'spam_prediction_time_seconds',
    'Time spent processing predictions',
    ['result']  # Added label dimension
)
ERROR_GAUGE = Gauge(
    'spam_errors_total',
    'Total prediction errors'
)

# Metric endpoint setup
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})



# ======================
# 2. MODEL LOADING WITH VALIDATION
# ======================
MODEL_DIR = os.getenv('MODEL_DIR', 'NoteBook')  # Use environment variable
MODEL_PATHS = {
    'model': os.path.join(MODEL_DIR, 'spam_model.pkl'),
    'tfidf': os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
}

def load_models():
    """Load models with proper pipeline handling"""
    try:
        # Verify files exist first
        for path in MODEL_PATHS.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load models
        pipeline = joblib.load(MODEL_PATHS['model'])
        tfidf = joblib.load(MODEL_PATHS['tfidf'])
        
        # Validate the pipeline contains a classifier
        if not hasattr(pipeline, 'named_steps'):
            raise ValueError("Loaded model is not a valid scikit-learn pipeline")
            
        # Get the classifier from the pipeline
        classifier = pipeline.named_steps.get('classifier')
        if classifier is None:
            raise ValueError("Pipeline doesn't contain a 'classifier' step")
        
        # Test feature dimensions (only if classifier has coef_)
        test_text = "validation text"
        X_test = tfidf.transform([test_text])
        
        if hasattr(classifier, 'coef_'):
            if classifier.coef_.shape[1] != X_test.shape[1]:
                raise ValueError(
                    f"Feature dimension mismatch! Model expects {classifier.coef_.shape[1]} "
                    f"features but got {X_test.shape[1]}"
                )
        
        print("✅ Models loaded successfully")
        return pipeline, tfidf
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print(f"Model path: {MODEL_PATHS['model']}")
        if os.path.exists(MODEL_PATHS['model']):
            loaded_model = joblib.load(MODEL_PATHS['model'])
            print(f"Model type: {type(loaded_model)}")
            if hasattr(loaded_model, 'named_steps'):
                print("Pipeline steps:", list(loaded_model.named_steps.keys()))
        return None, None

model, tfidf = load_models()

# ======================
# 3. CORE FUNCTIONALITY
# ======================
def clean_text(text):
    """Enhanced text cleaning that handles string input"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (add more cleaning steps as needed)
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    result = probability = error = None
    
    if request.method == 'POST' and model and tfidf:
        try:
            start_time = time.time()
            message = request.form.get('message', '')
            threshold = float(request.form.get('threshold', 0.5))
            
            # Clean the text BEFORE vectorization
            cleaned_text = clean_text(message)
            print(f"Cleaned text: {cleaned_text}")  # Debug print
            
            prob = model.predict_proba([cleaned_text])[0, 1]  # Pipeline handles vectorizing
            result = "SPAM" if prob > threshold else "HAM"
            probability = f"{prob:.4f}"
            
            # Debug output
            print(f"\n=== PREDICTION RESULTS ===")
            print(f"Original: {message}")
            print(f"Cleaned: {cleaned_text}")
            print(f"Probability: {probability}")
            print(f"Threshold: {threshold}")
            print(f"Result: {result}\n")
            
            # Record metrics
            duration = time.time() - start_time
            PREDICTION_COUNTER.labels(result=result).inc()
            PREDICTION_TIME.labels(result=result).observe(duration)
            
        except Exception as e:
            error = f"Error processing your message: {str(e)}"
            print(f"ERROR: {error}")
            ERROR_GAUGE.inc()
    
    return render_template('index.html',
        result=result,
        probability=probability,
        error=error,
        message=request.form.get('message', ''),
        threshold=request.form.get('threshold', '0.5')
    )
# ======================
# 4. HEALTH ENDPOINTS
# ======================
@app.route('/health')
def health_check():
    return {
        'status': 'ready' if model and tfidf else 'degraded',
        'model_loaded': bool(model),
        'vectorizer_loaded': bool(tfidf)
    }, 200 if model and tfidf else 503
# Model status
@app.route('/model-status')
def model_status():
    return {
        'model_loaded': bool(model),
        'tfidf_loaded': bool(tfidf),
        'model_type': str(type(model)) if model else 'None',
        'tfidf_type': str(type(tfidf)) if tfidf else 'None'
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# Run docker through cmd
# Run flask API http://localhost:5000
# Run grafana htgit tp://localhost:3000
# Run prometheus http://localhost:9090
