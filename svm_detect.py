import os
import cv2
import numpy as np
import joblib

def extract_features_svm(img_block):
    """
    Extract features as expected by the SVM model:
    normalized 3D HSV histogram (8x8x8 bins).
    """
    hsv = cv2.cvtColor(img_block, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

_svm_clf = None  # lazy-loaded singleton

def classify_color_svm(img_block):
    """
    Classify an image block using the provided SVM model (svm_color_model.pkl).
    """
    global _svm_clf
    if _svm_clf is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "svm_color_model.pkl")
        
        try:
            _svm_clf = joblib.load(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not load SVM model at {model_path}. Error: {e}")
            
    feature = extract_features_svm(img_block).reshape(1, -1)
    
    # Predict probabilities and get the best match
    classes = list(_svm_clf.classes_)
    probs = _svm_clf.predict_proba(feature)[0]
    pred_idx = np.argmax(probs)
    prediction = classes[pred_idx]
    
    # Map lowercase labels (red, blue, etc.) to App-standard Capitalized labels
    mapping = {
        'white': 'White', 'yellow': 'Yellow', 'orange': 'Orange',
        'red': 'Red', 'green': 'Green', 'blue': 'Blue'
    }
    return mapping.get(prediction.lower(), "White")
