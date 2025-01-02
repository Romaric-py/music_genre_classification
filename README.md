
# 🎵 Classification des Genres Musicaux 🎵

Ce projet utilise des techniques de machine learning pour classifier des fichiers audio dans différents genres musicaux. Il permet de prédire le genre musical d'une chanson à partir de ses caractéristiques audio extraites, telles que le chroma, la spectre, les MFCC, et plus encore. L'application est construite avec Streamlit et exploite des modèles préalablement entraînés pour effectuer les prédictions.

## 🛠 Prérequis

Le projet utilise Python 3.12.4 et les bibliothèques suivantes :

- **Streamlit** pour l'interface utilisateur.
- **Librosa** pour le traitement audio.
- **Scikit-learn** pour l'implémentation des modèles de machine learning.
- **Joblib** pour la sauvegarde des modèles.

Vous pouvez installer les dépendances requises via `pip` :

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` est inclus dans le projet et contient les versions nécessaires des bibliothèques.

## 🚀 Lancer l'application

Pour démarrer l'application Streamlit, exécutez le fichier `app.py` :

```bash
streamlit run app.py
```

L'application vous permettra de télécharger un fichier audio, d'afficher un signal audio et un spectrogramme, puis de prédire le genre musical en fonction des caractéristiques extraites du fichier.

## 🔧 Fonctionnalités

1. **Téléchargement de fichier audio** : Chargez un fichier `.wav` ou `.mp3` de votre choix.
2. **Affichage du signal audio** : Visualisez le signal audio sous forme d'onde.
3. **Spectrogramme** : Visualisez le spectrogramme Mel du fichier audio.
4. **Prédiction du genre** : Choisissez entre deux modèles (`LogisticRegression` ou `LinearSVC`) et obtenez une prédiction du genre musical. Le genre prédit est affiché avec un indicateur de confiance.
5. **Téléchargement des résultats** : Téléchargez les résultats de la prédiction sous forme de fichier texte.

## 🧑‍💻 Fonctionnement du modèle

Le modèle de prédiction fonctionne en extrayant plusieurs caractéristiques audio du fichier, telles que :

- Chroma (représente l'harmonie musicale).
- Spectral Centroid (mesure de la fréquence du spectre sonore).
- Spectral Bandwidth (étendue du spectre sonore).
- Mel-Frequency Cepstral Coefficients (MFCC) pour la description de la texture sonore.
- Tempo et taux de zéro-crossing, entre autres.

Ensuite, ces caractéristiques sont normalisées et passées à travers des modèles de machine learning préalablement entraînés (`Logistic Regression` et `Linear SVC`), qui renvoient le genre prédominant.

## 📄 Prédictions des genres

Les genres disponibles pour la prédiction sont :

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

### Exemple de prédiction :

Si vous chargez un fichier audio de genre blues, le modèle devrait prédire "blues" avec une certaine confiance, et vous verrez le genre affiché dans l'application.

## 👨‍💻 Contribuer

Si vous souhaitez contribuer au projet, vous pouvez :

1. Forker le projet.
2. Créer une branche pour votre fonctionnalité ou correction de bug.
3. Soumettre un pull request avec des explications sur votre modification.

## 📑 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Développé avec ❤️ par un passionné de musique et de Machine Learning.*
