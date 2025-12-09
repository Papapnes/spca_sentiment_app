# sentiment_utils.py (VERSION PRIORITÉ URGENCE & RAPIDE)
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modèle rapide et multilingue
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# ==========================================
# 1) Nettoyage du texte
# ==========================================
def nettoyer_texte(t):
    t = str(t).strip()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t


# ==========================================
# 2) Analyse du sentiment (super rapide)
# ==========================================
def analyser_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1).numpy()[0]
    label_index = probs.argmax()

    # Modèle retourne 5 classes (1 à 5 étoiles)
    if label_index <= 1:      # 1★ ou 2★
        sentiment = "negatif"
    elif label_index == 2:    # 3★
        sentiment = "neutre"
    else:                     # 4★ ou 5★
        sentiment = "positif"

    return sentiment, float(probs[label_index])


# ==========================================
# 3) Détection des thèmes SPCA
# ==========================================
THEMES = {
    "Maltraitance / Urgence animale": [
        "maltraitance", "cruel", "violence", "danger",
        "torture", "torturé", "blessé", "agonie"
    ],
    "Dons / Paiement": [
        "don", "paiement", "facture", "montant", "reçu",
        "carte", "taxe"
    ],
    "Adoption / Abandon": [
        "adoption", "adopter", "abandonné", "refuge"
    ],
    "Service / Communication": [
        "service", "réponse", "contact", "courriel", "téléphone",
        "attente", "expérience"
    ]
}

def detect_theme(text):
    txt = text.lower()
    for theme, keywords in THEMES.items():
        if any(k in txt for k in keywords):
            return theme
    return "Autre"


# ==========================================
# 4) Niveau d'urgence priorisé
# ==========================================
def niveau_urgence(row):
    txt = row["commentaire_clean"].lower()
    sent = row["sentiment"]

    if any(w in txt for w in ["maltraitance", "torture", "danger", "blessé", "agonie"]):
        return "URGENT"

    if sent == "negatif":
        return "À TRAITER"

    return "NORMAL"


# ==========================================
# 5) Pipeline principal
# ==========================================
def pipeline_analyse(df, col_commentaire="Commentaire"):

    df = df.copy()
    df[col_commentaire] = df[col_commentaire].fillna("")

    df["commentaire_clean"] = df[col_commentaire].apply(nettoyer_texte)

    sentiments = df["commentaire_clean"].apply(analyser_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["score"] = sentiments.apply(lambda x: x[1])

    df["theme"] = df["commentaire_clean"].apply(detect_theme)
    df["niveau_urgence"] = df.apply(niveau_urgence, axis=1)

    # Colonnes de priorité numérique
    def map_urgence(u):
        if u == "URGENT": return 1
        if u == "À TRAITER": return 2
        return 3

    def map_sentiment(s):
        if s == "negatif": return 1
        if s == "neutre": return 2
        return 3

    df["priorite_urgence"] = df["niveau_urgence"].apply(map_urgence)
    df["priorite_sentiment"] = df["sentiment"].apply(map_sentiment)

    # Tri final : URGENT → négatif → neutre → normal
    df = df.sort_values(by=["priorite_urgence", "priorite_sentiment"])

    return df
