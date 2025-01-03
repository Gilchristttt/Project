import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from gtts import gTTS
import os
from transformers import pipeline
#qa_pipeline = pipeline("question-answering",model="distilbert-base-uncased")
#qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2")
# Charger le contenu des fichiers
def charger_contenus(fichiers):
    contenus = {}
    for fichier in fichiers:
        try:
            if fichier.endswith(".pdf"):
                contenu = ""
                with pdfplumber.open(fichier) as pdf:
                    for page in pdf.pages:
                        contenu += page.extract_text()
            elif fichier.endswith(".txt"):
                with open(fichier, "r", encoding="utf-8") as f:
                    contenu = f.read()
            else:
                print(f"Format non supporté pour le fichier : {fichier}")
                continue
            contenus[fichier] = contenu.strip()  # Supprimer les espaces inutiles
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {fichier}: {e}")
    return contenus

# Calculer la pertinence de chaque fichier par rapport à la question
def calculer_score(question, contenus):
    tfidf = TfidfVectorizer()
    documents = list(contenus.values())
    tfidf_matrix = tfidf.fit_transform([question] + documents)
    scores = (tfidf_matrix[0] * tfidf_matrix[1:].T).toarray()
    return scores.argmax(), scores.max()

# Trouver le fichier le plus pertinent
def trouver_fichier_pertinent(question, contenus):
    index, score = calculer_score(question, contenus)
    fichier_pertinent = list(contenus.keys())[index]
    contenu_pertinent = list(contenus.values())[index]
    return fichier_pertinent, contenu_pertinent

# Lire du texte en audio
def lire_audio(texte, langue="fr"):
    tts = gTTS(texte, lang=langue)
    tts.save("output.mp3")
    os.system("start output.mp3")

# Mémoriser une question et une réponse
def memoire(question, reponse):
    try:
        with open("memoire.txt", "a", encoding="utf-8") as f:
            f.write(question + '\n')
            f.write(reponse + '\n')
    except Exception as e:
        print(f"Erreur lors de la sauvegarde dans la mémoire : {e}")

def reponse_naturelle(question, contenus):
    #try:
    fichier_pertinent, contexte = trouver_fichier_pertinent(question, contenus)
    resultat = qa_pipeline(question=question,context=contexte)
    print(f"Fichier pertinent : {fichier_pertinent}")
    #print(resultat["answer"])
    return resultat["answer"]
    #except Exception as e:
    #   print(f"Erreur lors de la génération de réponse : {e}")
    #    return "Je ne peux pas générer de réponse pour le moment."

# Obtenir une réponse pertinente
def reponse(question, contenus):
    fichier_pertinent, contenu_pertinent = trouver_fichier_pertinent(question, contenus)
    print(f"Fichier pertinent : {fichier_pertinent}")
    
    # Extraction des phrases pertinentes
    phrases = contenu_pertinent.split(".")
    reponses = [phrase.strip() for phrase in phrases if question.lower() in phrase.lower()]
    return reponses

# Charger les fichiers
fichiers = ["chapitre2.pdf", "chapitre4.pdf", "test.txt"]  
contenus = charger_contenus(fichiers)

# Bot interactif
while True:
    fichiers = ["chapitre2.pdf", "chapitre4.pdf", "test.txt"]  
    contenus = charger_contenus(fichiers)
    question = input("Posez une question (ou 'bye' pour quitter) : ").strip()
    #lire_audio(question)
    if question.lower() == "bye":
        lire_audio("Au revoir")
        break

    try:
        reponses = reponse(question, contenus)
        #reponses = reponse_naturelle(question, contenus)
        
        if reponses:
            #print(f"Réponse : {reponses}")
            #lire_audio(reponses)
            print(f"Réponse : {reponses[0]}")
            lire_audio(reponses[0])
        else:
            texte = "Je n'ai pas trouvé de réponse pertinente dans les fichiers."
            print(texte)
            lire_audio(texte)
            
            # Vérification dans la mémoire
            choix = input("Voulez-vous vérifier dans la mémoire ? (oui/non) : ").strip().lower()
            if choix == "oui":
                fichiers = ["memoire.txt"]
                contenus = charger_contenus(fichiers)
                reponses = reponse(question, contenus)
                #reponses = reponse_naturelle(question, contenus)
                
                if reponses:
                    #print(f"Réponse : {reponses}")
                    #lire_audio(reponses)
                    print(f"Réponse : {reponses[0]}")
                    lire_audio(reponses[0])
                else:
                    message = "Pas de réponse dans la mémoire non plus !"
                    print(message)
                    lire_audio(message)
                    
                    # Ajouter une nouvelle réponse à la mémoire
                    memoriser = input("Voulez-vous enregistrer cette question et sa réponse ? (oui/non) : ").strip().lower()
                    if memoriser == "oui":
                        reponse_user = input(f"Entrez la réponse à la question : {question} ")
                        memoire(question, reponse_user)
            else:
                continue
    except Exception as e:
        print(f"Erreur lors du traitement de la question : {e}")
        lire_audio("Une erreur s'est produite. Veuillez réessayer.")
