# sentiment_utils.py (VERSION PRO SPCA)
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator

# ======================
# 0) Modèle léger anglais (compatible Streamlit Cloud)
# ======================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# ======================
# 1) Détection automatique de la langue
# ======================
def detect_language(text):
    try:
        result = GoogleTranslator(source='auto', target='en').translate(text)
        return "fr" if result != text else "en"
    except:
        return "unknown"


# ======================
# 2) Traduction FR → EN
# ======================
def translate_to_en(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text


# ======================
# 3) Nettoyage du texte
# ======================
def nettoyer_texte(t):
    t = str(t).strip()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t


# ======================
# 4) Analyse du sentiment
# ======================
def analyser_sentiment(text):

    lang = detect_language(text)

    # Si français → traduction
    if lang == "fr":
        text_en = translate_to_en(text)
    else:
        text_en = text

    # Analyse sur modèle anglais
    inputs = tokenizer(text_en, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1).numpy()[0]
    neg, pos = probs

    # Classification
    if abs(pos - neg) < 0.10:
        label = "neutre"
    elif pos > neg:
        label = "positif"
    else:
        label = "negatif"

    return label, float(neg), float(pos), lang


# ======================
# 5) Thèmes SPCA avancés
# ======================

THEMES = {
    "Maltraitance / Urgence animale": [
        "maltraitance", "cruel", "violent", "torture", "torturé",
        "en danger", "blessé", "agonie", "animal errant",
        "négligé", "attaché", "mal nourri"
    ],
    "Dons / Paiement": [
        "don", "donateur", "donation", "paiement", "carte",
        "reçu", "recu", "facture", "montant", "taxe"
    ],
    "Adoption / Abandon": [
        "adoption", "adopter", "abandonné", "abandon", 
        "famille d'accueil", "refuge"
    ],
    "Euthanasie / Décision médicale": [
        "euthanasie", "endormir", "choix difficile", "vétérinaire"
    ],
    "Service / Communication": [
        "service", "réponse", "communication", "courriel",
        "téléphone", "attente", "expérience"
    ]
}

def detect_theme(text):
    text_l = text.lower()
    for theme, keywords in THEMES.items():
        if any(kw in text_l for kw in keywords):
            return theme
    return "Autre / À vérifier"


# ======================
# 6) Niveau d'urgence amélioré
# ======================

MOTS_CRITIQUES = [
    "maltraitance", "torture", "violence", "agonie",
    "en danger", "torturé", "blessé"
]

def niveau_urgence(row):

    txt = row["commentaire_clean"].lower()
    sent = row["sentiment"]
    theme = row["theme"]

    # Critique = URGENT
    if any(w in txt for w in MOTS_CRITIQUES):
        return "URGENT"

    if theme == "Maltraitance / Urgence animale":
        return "URGENT"

    # Négatif fort → priorité
    if sent == "negatif" and row["score_negatif"] > 0.60:
        return "PRIORITÉ MOYENNE"

    if sent == "negatif":
        return "À TRAITER"

    return "NORMAL"


# ======================
# 7) Génération d'une réponse courtoise automatisée
# ======================
def generer_reponse(sentiment, theme):

    if sentiment == "positif":
        return "Merci beaucoup pour votre message et votre soutien envers les animaux. Toute notre équipe apprécie votre bienveillance."

    if sentiment == "neutre":
        return "Merci pour votre commentaire. Nous avons bien pris note de votre retour et restons disponibles pour toute question."

    # Cas négatifs
    if theme == "Maltraitance / Urgence animale":
        return ("Merci de nous avoir signalé cette situation. "
                "Votre message a été transmis en priorité à notre équipe responsable du bien-être animal. "
                "Si un animal est en danger immédiat, contactez également les autorités compétentes.")

    if theme == "Dons / Paiement":
        return ("Merci pour votre message. Nous sommes désolés pour la situation concernant votre don. "
                "Notre équipe du service aux donateurs vous contactera rapidement pour résoudre le problème.")

    if theme == "Service / Communication":
        return ("Merci de votre message. Nous sommes désolés si votre expérience n’a pas été satisfaisante. "
                "Nous nous engageons à améliorer nos communications.")

    return ("Merci pour votre commentaire. Nous sommes désolés pour la situation et allons faire le suivi nécessaire.")


# ======================
# 8) Pipeline complet
# ======================
def pipeline_analyse(df, col_commentaire="Commentaire"):

    df = df.copy()
    df[col_commentaire] = df[col_commentaire].fillna("")

    df["commentaire_clean"] = df[col_commentaire].apply(nettoyer_texte)

    sentiments = df["commentaire_clean"].apply(analyser_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["score_negatif"] = sentiments.apply(lambda x: x[1])
    df["score_positif"] = sentiments.apply(lambda x: x[2])
    df["langue"] = sentiments.apply(lambda x: x[3])

    df["theme"] = df["commentaire_clean"].apply(detect_theme)
    df["niveau_urgence"] = df.apply(niveau_urgence, axis=1)

    df["reponse_proposee"] = df.apply(
        lambda r: generer_reponse(r["sentiment"], r["theme"]),
        axis=1
    )

    return df
