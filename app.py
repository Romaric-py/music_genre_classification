import streamlit as st
import librosa
import json
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from model_inference import predict_genre, predict


# Fonctions
@st.cache_data
def load_file(audio_file):
    return librosa.load(uploaded_file, sr=22050)

@st.cache_data
def display_signal(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4")
    ax.set_title("Signal Audio", fontsize=15)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

@st.cache_data
def display_spectogram(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='plasma')
    ax.set_title("Spectrogramme Mel", fontsize=15)
    st.pyplot(fig)
    
@st.cache_data
def make_prediction(y, chosen_model):
    # Modèle choisi
    use_model = ('log' if chosen_model=='LogisticRegression' else 'svc')
    # Prédire le genre
    y_trimmed, _ = librosa.effects.trim(y)
    return predict_genre(y_trimmed, sr, use_model=use_model), predict(y_trimmed, sr, use_model=use_model)




# Titre de l'application
st.balloons()
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #00c4ff;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        color: #dddddd;
        margin-top: 0;
    }
    </style>
    <h1 class="main-title">🎵 Classification des Genres Musicaux 🎵</h1>
    <h2 class="subtitle">Transformez votre musique en un genre défini !</h2>
    """, unsafe_allow_html=True)

# La SideBar
chosen_model = st.sidebar.selectbox("Choisir un modèle pour la prédiction", options=['LogisticRegression', 'LinearSVC'])

# Téléchargement du fichier audio
uploaded_file = st.file_uploader("Upload un fichier audio pour prédiction (minimum 30 secondes):", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        # Charger le fichier audio
        y, sr = load_file(uploaded_file)
        
        # Lecture du fichier
        st.audio(uploaded_file)

        # Visualisation du signal audio
        st.subheader("🎶 Visualisation du Signal Audio")
        display_signal(y)

        # Visualisation du spectrogramme
        st.subheader("🎛️ Spectrogramme")
        display_spectogram(y)

        with st.spinner("Analyse du caractéristiques de l'audio"):
            (predicted_genre, predictions), global_pred = make_prediction(y, chosen_model)
                    
            # Résultat
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(45deg, #1e3c72, #2a5298); color: white; margin: 1rem 0;">
                <h2>🎤 Genre Prédit :</h2>
                <h1 style="font-size: 3rem;"> {predicted_genre}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            
        # Info sur les prédictions des modèles
        st.info(f"De manière globale, cet audio est proche du {global_pred}")
        st.info("Les genres prédictibles sont: *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock* avec une confiance de 75%.")
        st.info("Les résultats peuvent varier selon le modèle utlisé. Ayez aussi en tête qu'une même chanson peut inclure plusieurs genres.")
        st.markdown(f"Le modèle utilisé est **:blue[{chosen_model}]**.")
            
        st.write(predictions)
        # Bouton de téléchargement des résultats
        st.download_button(
            label="📥 Télécharger Résultats",
            data=str(predictions),
            file_name="resultat_genre.txt",
        )
        st.success("Analyse terminée")

    except Exception as e:
        st.error(f"Un erreur est survenue")
        with st.expander("Plus de détails"):
            st.markdown(f'\n{e}')

# Note de bas de page
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        text-align: center;
        font-size: 0.8rem;
        color: #bbbbbb;
        width: 100%;
    }
    </style>
    <div class="footer">
        Développé avec ❤️ par un passionné de musique et de Machine Learning.
    </div>
    """,
unsafe_allow_html=True)
