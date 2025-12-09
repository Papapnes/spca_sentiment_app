# sentiment_utils.py
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator

# Petit modèle anglais compatible Streamlit Cloud
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
id2label = {0: "negatif", 1: "positif"}  # binaire


# ================
# Traduction FR → EN
# ================
def traduire_en(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  # si erreur → on garde original


# ======================
# 1) Nettoyage du texte
# ======================
def nettoyer_texte(t):
    t = str(t).strip()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t


# ======================
# 2) Analyse du sentiment
# ======================
def analyser_sentiment(text):

    # Traduction automatique FR → EN
    text_en = traduire_en(text)

    inputs = tokenizer(text_en, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1).numpy()[0]

    # Label du modèle
    label_en = "positif" if probs[1] > probs[0] else "negatif"

    # Ajouter neutre si scores proches
    if abs(probs[1] - probs[0]) < 0.15:
        label_en = "neutre"

    return label_en, float(probs[0]), float(probs[1])


# LE RESTE DE TON CODE RESTE IDENTIQUE
# (détection de thème et niveau d'urgence)
# ============================================================
THEMES = {
    "Dons / Paiement": [
        "don", "donation", "reçu", "recu", "reçu d'impôt", "taxe",
        "carte de crédit", "paiement", "facture", "montant",
        "annulation", "remboursement"
    ],
    "Adoption / Abandon": [
        "adoption", "adopter", "adopté", "abandonné", "abandon"
    ],
    "Maltraitance / Urgence animale": [
        "maltraitance", "cruel", "violent", "blessé", "en danger",
        "urgence", "attaché", "négligé", "torturé"
    ],
    "Service / Communication": [
        "service", "réponse", "contact", "email", "courriel",
        "téléphone", "attente", "retard"
    ]
}

def detect_theme(txt):
    txt_l = txt.lower()
    for theme, keywords in THEMES.items():
        if any(kw in txt_l for kw in keywords):
            return theme
    return "Autre / À vérifier"


MOTS_URGENCE_CRITIQUE = [
    "maltraitance", "torture", "torturé", "en danger",
    "blessé", "agonie", "urgence", "cruel", "violence"
]

def niveau_urgence(row):
    txt = row["commentaire_clean"].lower()
    sent = row["sentiment"]
    theme = row["theme"]

    if any(m in txt for m in MOTS_URGENCE_CRITIQUE) or theme == "Maltraitance / Urgence animale":
        return "URGENT"

    if sent == "negatif" and row["score_negatif"] >= 0.60:
        return "PRIORITÉ MOYENNE"

    if sent == "negatif":
        return "À TRAITER"

    return "NORMAL"


def pipeline_analyse(df, col_commentaire="Commentaire"):
    df = df.copy()
    df[col_commentaire] = df[col_commentaire].fillna("")

    df["commentaire_clean"] = df[col_commentaire].apply(nettoyer_texte)

    sentiments = df["commentaire_clean"].apply(analyser_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["score_negatif"] = sentiments.apply(lambda x: x[1])
    df["score_positif"] = sentiments.apply(lambda x: x[2])

    df["theme"] = df["commentaire_clean"].apply(detect_theme)
    df["niveau_urgence"] = df.apply(niveau_urgence, axis=1)

    return df
