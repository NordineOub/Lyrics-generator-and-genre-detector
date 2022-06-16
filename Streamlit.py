import streamlit as st
import joblib
from PIL import Image
import numpy as np
from markovchain import JsonStorage
from markovchain.text import MarkovText, ReplyMode

# -----------------------------------  Page 1
@st.cache
def import_AI_model(): 
        model_from_joblib = joblib.load('pipe_svc_tf_0658.pkl')
        return model_from_joblib

def findGenre(df):
    df =[df]
    model =import_AI_model()
    st.write(str(model.predict(df)))

# -----------------------------------   Page 2

def import_markov_model(genre):
    lyrics = []
    markov = MarkovText.from_file('markov_model.json')
    st.write('Paroles de lyrics :\n\n')
    paroles =markov(max_length=190, reply_to = genre)
    for mots in paroles:
        if (mots ==','):
            lyrics.append('\n')
        else :
            lyrics.append(mots)
    lyrics =''.join(lyrics)
    st.write(lyrics)

# -----------------------------------  Page 3
def savoirplus():
    pie = Image.open('output_Dataset.png')
    st.image(pie, caption='Proportion du dataset en fonction du genre')
    st.write("\nVoici la répartition de notre dataset. Il se compose d'un nombre inégal de lyrics de différents genre musicaux (ce qui a compliqué la tâche d'apprentissage du fait de l'hétérogéinité du dataset\n\n")
    accuracy_pipe= Image.open("accuracy_models.png")
    st.image(accuracy_pipe,caption="Accuracy des différents moodèles testés")
    st.write("\nNous avons testé ces différents modèles dans le but de trouver celui avec la meilleure accuracy. Voici le résultat")
    frequence=Image.open("Frequency_distribution_words.png")
    st.image(frequence, caption ="Fréquence des mots pour chaque musique")

# ----------------------------------- Sidebar

def selectTool():
    add_selectbox = st.sidebar.radio(
    "Méthode :",
    ("Reconnaissance Genre", "Nouveau Lyrics","En savoir plus sur le Dataset")  
)
    if (add_selectbox== "Reconnaissance Genre"):
        df =st.text_input(' DETECTION DE GENRE : Veuillez écrire une partie des lyrics du morceau',"")
        findGenre(df)
    elif(add_selectbox== "Nouveau Lyrics"):
        lyric =st.text_input('GENERATEUR DE LYRICS : Veuillez écrire le début de vos lyrics du morceau',"")
        if (lyric == 0): 
             return None 
        else: 
            import_markov_model(lyric)
    elif(add_selectbox=="En savoir plus sur le Dataset"):
        savoirplus()
        df = None
        return add_selectbox





st.title ('Machine learning II : Détection de genres musicaux et écriture automatique de Lyrics')
st.header ('Par Nour-Eddine OUBENAMI\n\n')
selectTool()
