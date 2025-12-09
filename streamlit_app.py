# streamlit_app.py
import streamlit as st
import pandas as pd
from sentiment_utils import pipeline_analyse

st.set_page_config(
    page_title="SPCA - Analyse des commentaires",
    layout="wide"
)

st.markdown(
    "<h2 style='color:#ae0f27;'>🐾 SPCA – Analyse automatisée des commentaires</h2>",
    unsafe_allow_html=True
)

st.write("Téléverse un fichier CSV contenant les commentaires. L'application analyse :")
st.write("- le sentiment (positif / neutre / négatif)")
st.write("- le thème (dons, adoption, maltraitance, etc.)")
st.write("- le niveau d'urgence")
st.write("- une réponse courtoise automatique")

uploaded_file = st.file_uploader("📂 Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)
    st.success(f"Fichier chargé avec {len(df_raw)} lignes.")

    col_comment = st.selectbox(
        "Choisir la colonne des commentaires :",
        options=df_raw.columns.tolist()
    )

    if st.button("🚀 Lancer l'analyse"):

        with st.spinner("Analyse en cours..."):
            df_result = pipeline_analyse(df_raw, col_commentaire=col_comment)

        # Statistiques
        st.subheader("📊 Résumé")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Commentaires", len(df_result))
        col2.metric("Urgents", (df_result["niveau_urgence"] == "URGENT").sum())
        col3.metric("Négatifs", (df_result["sentiment"] == "negatif").sum())
        col4.metric("Positifs", (df_result["sentiment"] == "positif").sum())

        # Filtres sécurisés
        st.subheader("🎯 Filtres")
        urgences_options = df_result["niveau_urgence"].unique().tolist()
        default_urgences = [u for u in ["URGENT", "PRIORITÉ MOYENNE"] if u in urgences_options]

        urgences = st.multiselect(
            "Filtrer par niveau d'urgence",
            options=urgences_options,
            default=default_urgences
        )

        themes_options = df_result["theme"].unique().tolist()
        themes = st.multiselect(
            "Filtrer par thème",
            options=themes_options,
            default=themes_options
        )

        df_filtered = df_result.copy()
        if urgences:
            df_filtered = df_filtered[df_filtered["niveau_urgence"].isin(urgences)]
        if themes:
            df_filtered = df_filtered[df_filtered["theme"].isin(themes)]

        # Tableau résultats
        st.subheader("📄 Commentaires analysés")
        st.dataframe(
            df_filtered[[
                col_comment, "sentiment", "score_negatif",
                "score_positif", "theme", "niveau_urgence",
                "reponse_proposee"
            ]],
            use_container_width=True
        )

        # Export CSV complet
        csv = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "💾 Télécharger tous les résultats (CSV)",
            data=csv,
            file_name="commentaires_spca_analyses.csv",
            mime="text/csv"
        )

else:
    st.info("En attente d’un fichier...")
