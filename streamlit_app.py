# streamlit_app.py
import streamlit as st
import pandas as pd
from sentiment_utils import pipeline_analyse

st.set_page_config(
    page_title="Analyse des commentaires",
    layout="wide"
)

st.markdown(
    "<h2 style='color:#ae0f27;'> 💬 Analyse des commentaires 💬 </h2>",
    unsafe_allow_html=True
)

st.write("Téléverse un fichier CSV contenant les commentaires. L'application analyse :")
st.write("- les sentiments négatifs et urgents")
st.write("- les thèmes associés (maltraitance, dons, adoption, etc.)")
st.write("- les niveaux d'urgence (URGENT, À TRAITER, NORMAL)")
st.write("- tri automatique du plus urgent au moins urgent")

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
        col1, col2, col3 = st.columns(3)
        col1.metric("Total commentaires", len(df_result))
        col2.metric("Urgents", (df_result["niveau_urgence"] == "URGENT").sum())
        col3.metric("Négatifs", (df_result["sentiment"] == "negatif").sum())

        # Filtres
        st.subheader("🎯 Filtres")
        urgences_options = df_result["niveau_urgence"].unique().tolist()
        urgences = st.multiselect(
            "Filtrer par urgence",
            options=urgences_options,
            default=urgences_options
        )

        themes_options = df_result["theme"].unique().tolist()
        themes = st.multiselect(
            "Filtrer par thème",
            options=themes_options,
            default=themes_options
        )

        # Application des filtres
        df_filtered = df_result.copy()
        if urgences:
            df_filtered = df_filtered[df_filtered["niveau_urgence"].isin(urgences)]
        if themes:
            df_filtered = df_filtered[df_filtered["theme"].isin(themes)]

        # Affichage final trié automatiquement
        st.subheader("📄 Commentaires priorisés")

        st.dataframe(
            df_filtered[[
                col_comment,
                "sentiment",
                "score",
                "theme",
                "niveau_urgence",
                "priorite_urgence"
            ]],
            use_container_width=True
        )

        # Export CSV
        csv = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "💾 Télécharger tous les résultats (CSV)",
            data=csv,
            file_name="commentaires_priorises_spca.csv",
            mime="text/csv"
        )

else:
    st.info("En attente d’un fichier…")
