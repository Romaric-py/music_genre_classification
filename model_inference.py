import joblib
import numpy as np
from extraction import extract_features, feature_names
from sklearn.preprocessing import StandardScaler
from collections import Counter
from multiprocessing.pool import ThreadPool
import multiprocessing as mp

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
    return [y[i: i+thirty_sec] 
            for i in range(0, len(y), thirty_sec)
            if len(y[i: i+thirty_sec]) > thirty_sec//2]

def predict_genre(y, sr, *, use_model='log'):
    # splitter y
    ys = split(y, sr)
    # faire les prédictions en parallélisant l'analyse des segments 
    with ThreadPool(mp.cpu_count()) as pool:
        predictions = pool.starmap(predict, [(y, sr, False, use_model) for y in ys])
    # predictions = [predict(x, sr, use_model=use_model) for x in ys]
    # sélectionner la classe prépondérante
    counter = Counter(predictions)
    return counter.most_common(1)[0][0], predictions
