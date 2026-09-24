"""
Bovine SNP Platform - version minimale de test.
Si cette version marche, on ajoutera les modules progressivement.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Bovine SNP Platform",
    page_icon="DNA",
    layout="wide",
)

st.title("Bovine SNP Platform - Version de test")
st.caption("Si tu vois cette page, le fichier compile correctement.")

st.sidebar.header("Donnees")

n_ind = st.sidebar.number_input("Individus", 20, 500, 100, 10)
n_snp = st.sidebar.number_input("SNPs", 50, 5000, 500, 50)

if st.sidebar.button("Generer un jeu de demo"):
    rng = np.random.default_rng(42)
    gt = rng.integers(0, 3, size=(int(n_ind), int(n_snp)))
    df = pd.DataFrame(gt[:, :20],
                      columns=["SNP_" + str(i) for i in range(20)])
    st.session_state["gt"] = gt
    st.session_state["df"] = df
    st.success("Genere : " + str(gt.shape[0]) + " x " + str(gt.shape[1]))

if "gt" in st.session_state:
    st.subheader("Apercu des donnees")
    st.write("Dimensions :", st.session_state["gt"].shape)
    st.dataframe(st.session_state["df"].head(10))

    tab1, tab2 = st.tabs(["Histogramme", "QC basique"])

    with tab1:
        fig = px.histogram(
            x=st.session_state["gt"].ravel(),
            nbins=3,
            title="Distribution des genotypes (0, 1, 2)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        gt = st.session_state["gt"]
        st.write("Frequences alleles:")
        st.write("SNP 0 :", float(gt[:, 0].mean()))
        st.write("SNP 1 :", float(gt[:, 1].mean()))
        st.write("SNP 2 :", float(gt[:, 2].mean()))
