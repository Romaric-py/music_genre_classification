import joblib
import librosa
from extraction import extract_features, feature_names
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Charger le pipeline
LOG_MODEL_PATH = 'output/log_clf_wl.joblib'
SVC_MODEL_PATH = "output/linSVC_wl.joblib"
LABEL_ENCODER_PATH = 'output/label_encoder.joblib'

model_log = joblib.load(LOG_MODEL_PATH)
model_svc = joblib.load(SVC_MODEL_PATH)
lb_encoder = joblib.load(LABEL_ENCODER_PATH)
    

def predict(y, sr, labels=False, use_model='log'):
    # Extraction des carcatéristiques
    features = extract_features(y, sr)[1:] # on ignore la longueur
    features = features.reshape(1, -1)
    # choisir le modèle
    model = (model_log if use_model=='log' else model_svc)
    # Faire la prédiction
    predictions = model.predict(features)
    if labels:
        predictions = lb_encoder.inverse_transform(predictions)
    prediction = predictions[0]
    return prediction

def split(y, sr):
    thirty_sec = 30 * sr
    n_chunks = len(y) // (thirty_sec) + 1
    return [y[thirty_sec*i: thirty_sec*(i+1)] 
            for i in range(n_chunks)
            if len(y[thirty_sec*i: thirty_sec*(i+1)]) > thirty_sec//2]

def predict_genre(y, sr, *, use_model='log'):
    # splitter y
    ys = split(y, sr)
    # faire les prédictions
    predictions = [predict(x, sr, use_model=use_model) for x in ys]
    # sélectionner la classe prépondérante
    counter = Counter(predictions)
    return counter.most_common(1)[0][0], predictions
