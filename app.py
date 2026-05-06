import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

## PAGE SET UP
st.set_page_config(page_title="Unsupervised Machine Learning Tool", layout="wide")
st.title("Unsupervised Machine Learning Tool")
st.markdown("Upload a dataset, experiment with hyperparameters, and observe how these affect model training and performance.")

## SIDEBAR
with st.sidebar:

    st.header("Upload Dataset")

    dataset = st.file_uploader("Upload a CSV file", type="csv")
    df = None
    if dataset:
        df = pd.read_csv(dataset)
    else:
        st.info("Upload a CSV file.")

    if df is not None:
        st.divider()
        st.header("Choose Variables")

        all_cols = df.columns.tolist()
        feature_cols = st.multiselect(
            "Feature variables (columns to include in analysis)",
            all_cols,
            default=all_cols[:2]
        )

        if not feature_cols:
            st.error("Select one or more feature variables.")
            st.stop()

        st.divider()
        st.header("Choose a Model")

        model_name = st.selectbox(
            "Model",
            ["K-Means Clustering", "PCA"],
            help=("K-Means Clustering: groups data into k clusters based on similarity.\n"
                  "PCA (Principal Component Analysis): reduces dimensions while preserving as much variance as possible.")
        )

        st.divider()
        st.header("Tune Hyperparameters")
        st.caption("Tune for model testing and performance.")

        random_state = st.number_input("Random seed", value=100, step=1)

        model_params: dict = {}

        if model_name == "K-Means Clustering":

            model_params["n_clusters"] = st.slider("Number of clusters (k)", 2, 15, 3)
            model_params["init"] = st.selectbox("Initialization method", ["k-means++", "random"])
            model_params["n_init"] = st.slider("Number of initializations (n_init)", 1, 20, 10)
            model_params["max_iter"] = st.slider("Max iterations", 50, 500, 300, step=50)
            model_params["random_state"] = int(random_state)

        elif model_name == "PCA":

            numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            max_components = max(len(numeric_cols), 1)

            model_params["n_components"] = st.slider(
                "Number of components",
                1, max_components, min(2, max_components)
            )

            model_params["whiten"] = st.checkbox("Whiten", value=False)
            model_params["random_state"] = int(random_state)

## MAIN PANEL
if df is None:
    st.info("Upload a dataset to perform analysis.")
    st.stop()

with st.expander("Quick Dataset Preview", expanded=True):
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("**Column Names:**")
        dtype_df = pd.DataFrame({
            "column": df.columns.tolist(),
            "type": [str(dt) for dt in df.dtypes]
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")

    with right_col:
        st.markdown("**First 12 rows:**")
        st.dataframe(df.head(12), use_container_width=True)

    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

## TRAINING
with st.spinner("Running the model"):

    try:
        all_var = df[feature_cols].copy()

        # ✅ FIXED encoding (robust for all pandas versions)
        for col in all_var.columns:
            if not pd.api.types.is_numeric_dtype(all_var[col]):
                all_var[col] = OrdinalEncoder().fit_transform(all_var[[col]])

        old_len = len(all_var)
        all_var = all_var.dropna()
        num_dropped = old_len - len(all_var)

        if num_dropped == 1:
            st.caption("1 row with missing values was dropped before training.")
        elif num_dropped > 1:
            st.caption(f"{num_dropped} rows with missing values were dropped before training.")

        X = all_var.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ## K-MEANS
        if model_name == "K-Means Clustering":

            model = KMeans(**model_params)
            labels = model.fit_predict(X_scaled)

            inertia = model.inertia_
            sil_score = silhouette_score(X_scaled, labels)

            st.subheader("Model Performance: K-Means Clustering")
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters (k)", model_params["n_clusters"])
            col2.metric("Inertia", f"{inertia:.2f}")
            col3.metric("Silhouette Score", f"{sil_score:.4f}")

        ## PCA
        elif model_name == "PCA":

            numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_features) < 2:
                st.error("PCA requires at least 2 numeric feature columns.")
                st.stop()

            X_pca = all_var[numeric_features].values
            X_pca_scaled = StandardScaler().fit_transform(X_pca)

            model = PCA(**model_params)
            X_transformed = model.fit_transform(X_pca_scaled)

            evr = model.explained_variance_ratio_
            cumulative_var = np.cumsum(evr)

            st.subheader("Model Performance: PCA")
            col1, col2, col3 = st.columns(3)
            col1.metric("Components", model_params["n_components"])
            col2.metric("Variance Explained", f"{cumulative_var[-1]*100:.2f}%")
            col3.metric("PC1 Variance", f"{evr[0]*100:.2f}%")

    except Exception as e:
        st.error(f"Model failed: {e}")
        st.exception(e)
