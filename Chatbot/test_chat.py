from transformers import pipeline

# Charger le pipeline de question-réponse
#qa_pipeline = pipeline("question-answering",model="distilbert-base-uncased")
qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2")


context = """
La capitale de la France est Paris. Elle est connue pour sa riche histoire, ses monuments célèbres comme la Tour Eiffel,
et sa gastronomie. La France est également une puissance mondiale dans les domaines de la culture, de l'économie et de la politique.
"""
contenu=""
with open("test.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

question = "le plus grand océan du monde ?"

# Utiliser le pipeline pour répondre à la question
result = qa_pipeline(question=question, context=contenu)

# Afficher la réponse
print(f"Question : {question}")
print(f"Réponse : {result['answer']}")
print(f"Score de confiance : {result['score']}")
