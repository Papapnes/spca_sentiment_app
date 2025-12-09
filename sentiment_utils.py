# sentiment_utils.py (VERSION ULTRA RAPIDE)
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modèle très rapide et multilingue
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def nettoyer_texte(t):
    t = str(t).strip()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def analyser_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1).numpy()[0]
    label_index = probs.argmax()

    # Modèle donne 1 à 5 étoiles
    if label_index <= 1:
        sentiment = "negatif"
    elif label_index == 2:
        sentiment = "neutre"
    else:
        sentiment = "positif"

    return sentiment, float(probs[label_index])


THEMES = {
    "Maltraitance / Urgence animale": [
        "maltraitance", "cruel", "violence", "blessé", "danger",
        "négligé", "torturé", "abandon"
    ],
    "Dons / Paiement": [
        "don", "paiement", "facture", "montant", "taxe", "carte"
    ],
    "Adoption / Abandon": [
        "adoption", "adopter", "abandonné", "refuge"
    ],
    "Service / Communication": [
        "service", "réponse", "contact", "courriel", "attente"
    ]
}

def detect_theme(text):
    txt = text.lower()
    for theme, keywords in THEMES.items():
        if any(k in txt for k in keywords):
            return theme
    return "Autre"


def niveau_urgence(row):
    txt = row["commentaire_clean"].lower()
    sent = row["sentiment"]
    theme = row["theme"]

    if any(w in txt for w in ["maltraitance", "torture", "blessé", "danger"]):
        return "URGENT"

    if theme == "Maltraitance / Urgence animale":
        return "URGENT"

    if sent == "negatif":
        return "À TRAITER"

    return "NORMAL"


def pipeline_analyse(df, col_commentaire="Commentaire"):
    df = df.copy()
    df[col_commentaire] = df[col_commentaire].fillna("")
    df["commentaire_clean"] = df[col_commentaire].apply(nettoyer_texte)

    sentiments = df["commentaire_clean"].apply(analyser_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["score"] = sentiments.apply(lambda x: x[1])

    df["theme"] = df["commentaire_clean"].apply(detect_theme)
    df["niveau_urgence"] = df.apply(niveau_urgence, axis=1)

    return df
