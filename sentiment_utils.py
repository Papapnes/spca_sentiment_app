# sentiment_utils.py
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle de sentiment multilingue
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
id2label = {0: "negatif", 1: "neutre", 2: "positif"}


# =========================
# 1) Nettoyage du texte
# =========================
def nettoyer_texte(t):
    t = str(t).strip()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t


# =========================
# 2) Analyse du sentiment
# =========================
def analyser_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutre", 0.0, 1.0, 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1).numpy()[0]
    label = id2label[int(probs.argmax())]

    return label, float(probs[0]), float(probs[1]), float(probs[2])


# =========================
# 3) Détection du thème
# =========================

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


# =========================
# 4) Niveau d'urgence
# =========================

MOTS_URGENCE_CRITIQUE = [
    "maltraitance", "torture", "torturé", "en danger",
    "blessé", "agonie", "urgence", "cruel", "violence"
]

def niveau_urgence(row):
    txt = row["commentaire_clean"].lower()
    sent = row["sentiment"]
    theme = row["theme"]

    # Cas toujours URGENT
    if any(m in txt for m in MOTS_URGENCE_CRITIQUE) or theme == "Maltraitance / Urgence animale":
        return "URGENT"

    # Donateur très fâché
    if sent == "negatif" and row["score_negatif"] >= 0.60:
        return "PRIORITÉ MOYENNE"

    # Autres commentaires négatifs
    if sent == "negatif":
        return "À TRAITER"

    return "NORMAL"


# =========================
# 5) Pipeline complet
# =========================
def pipeline_analyse(df, col_commentaire="Commentaire"):
    df = df.copy()
    df[col_commentaire] = df[col_commentaire].fillna("")

    # Nettoyage
    df["commentaire_clean"] = df[col_commentaire].apply(nettoyer_texte)

    # Sentiment
    sentiments = df["commentaire_clean"].apply(analyser_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["score_negatif"] = sentiments.apply(lambda x: x[1])
    df["score_neutre"] = sentiments.apply(lambda x: x[2])
    df["score_positif"] = sentiments.apply(lambda x: x[3])

    # Thème
    df["theme"] = df["commentaire_clean"].apply(detect_theme)

    # Urgence
    df["niveau_urgence"] = df.apply(niveau_urgence, axis=1)

    return df
