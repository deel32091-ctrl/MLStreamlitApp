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
    # Ask to upload a CSV file and read it once it is uploaded
    if dataset:
        df = pd.read_csv(dataset)
    else:
        st.info("Upload a CSV file.")

    # Once the dataset is uploaded, reveal the widgets
    if df is not None:
        st.divider()
        st.header("Choose Variables")
        # Allow picking as many feature variables as you want (no target needed for unsupervised)
        all_cols = df.columns.tolist()
        feature_cols = st.multiselect(
            "Feature variables (columns to include in analysis)",
            all_cols,
            default=all_cols[:2])
        if not feature_cols:
            st.error("Select one or more feature variables.")
            st.stop()

        st.divider()
        st.header("Choose a Model")
        # Choose from the two models with a help section so the user knows what they are picking
        model_name = st.selectbox(
            "Model",
            ["K-Means Clustering", "PCA"],
            help=("K-Means Clustering: groups data into k clusters based on similarity.\n"
                  "PCA (Principal Component Analysis): reduces dimensions while preserving as much variance as possible."))

        st.divider()
        st.header("Tune Hyperparameters")
        st.caption("Tune for model testing and performance.")

        # Since results can vary by random initialization, allow the user to change the seed
        random_state = st.number_input("Random seed", value=100, step=1,
            help="Controls random initialization. Change this to test stability.")

        # Model hyperparameter section — each model has different hyperparameters described in help sections
        model_params: dict = {}

        if model_name == "K-Means Clustering":

            model_params["n_clusters"] = st.slider(
                "Number of clusters (k)",
                2, 15, 3,
                help="The number of clusters to form. Use the Elbow Plot tab to help choose the best k.")

            model_params["init"] = st.selectbox(
                "Initialization method",
                ["k-means++", "random"],
                help="k-means++: smart initialization that speeds up convergence and avoids poor results.\nrandom: picks random starting centroids.")

            model_params["n_init"] = st.slider(
                "Number of initializations (n_init)",
                1, 20, 10,
                help="How many times the algorithm runs with different random seeds. The best result is kept. Higher = more stable but slower.")

            model_params["max_iter"] = st.slider(
                "Max iterations",
                50, 500, 300,
                step=50,
                help="Maximum number of iterations per run before stopping. Increase if the model is not converging.")

            model_params["random_state"] = int(random_state)

        elif model_name == "PCA":

            # Dynamically set the max components to the number of numeric feature columns selected
            numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            max_components = max(len(numeric_cols), 2)

            model_params["n_components"] = st.slider(
                "Number of components",
                1, max_components, min(2, max_components),
                help="How many principal components to retain. More components = more variance explained. Use the Cumulative Variance tab to find the right number.")

            model_params["whiten"] = st.checkbox(
                "Whiten",
                value=False,
                help="Scales each component to unit variance. Useful when features have very different scales, or before feeding PCA output into another model.")

            model_params["random_state"] = int(random_state)

## MAIN PANEL
if df is None:
    st.info("Upload a dataset to perform analysis.")
    st.stop()

# Dataset preview
with st.expander("Quick Dataset Preview", expanded=True):
    left_col, right_col = st.columns([1, 2])

    with left_col:
        # Left column shows column names and their data types
        # FIX: convert dtype objects to strings so pyarrow can serialize them
        st.markdown("**Column Names:**")
        dtype_df = (df.dtypes
                      .rename("type")
                      .reset_index()
                      .rename(columns={"index": "column"}))
        dtype_df["type"] = dtype_df["type"].astype(str)
        st.dataframe(dtype_df, width="stretch", hide_index=True)

        # Underneath, the dimensions of the dataset
        st.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")

    with right_col:
        # Right column shows the first 12 rows
        st.markdown("**First 12 rows:**")
        st.dataframe(df.head(12), width="stretch")

    # Below both columns: descriptive statistics
    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df.describe(), width="stretch")

## TRAINING
# Use st.spinner so the model runs immediately whenever a change is made in the sidebar
with st.spinner("Running the model"):

    try:
        # Grab only the selected feature columns
        all_var = df[feature_cols].copy()

        # FIX: include "str" alongside "object"/"category" for pandas 3 compatibility
        for col in all_var.select_dtypes(include=["object", "category", "str"]).columns:
            all_var[col] = OrdinalEncoder().fit_transform(all_var[[col]])

        # Drop rows with missing values — missing data is a separate problem
        old_len = len(all_var)
        all_var = all_var.dropna()
        num_dropped = old_len - len(all_var)
        if num_dropped == 1:
            st.caption("1 row with missing values was dropped before training.")
        elif num_dropped > 1:
            st.caption(f"{num_dropped} rows with missing values were dropped before training.")

        X = all_var.values

        # Scale features so distance-based and variance-based models work properly
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ## K-MEANS CLUSTERING SECTION

        if model_name == "K-Means Clustering":

            model = KMeans(**model_params)
            labels = model.fit_predict(X_scaled)

            # Calculate the evaluation metrics
            inertia   = model.inertia_
            sil_score = silhouette_score(X_scaled, labels)

            # Present the metrics below the dataset preview
            st.subheader("Model Performance: K-Means Clustering")
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters (k)",      model_params["n_clusters"])
            col2.metric("Inertia",           f"{inertia:.2f}")
            col3.metric("Silhouette Score",  f"{sil_score:.4f}")

            # Explain what a good score looks like
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Clusters (k)** | The number of clusters you chose. Use the Elbow Plot to find the best k. |
| **Inertia** | Sum of squared distances from each point to its cluster center. Lower = tighter, more compact clusters. Compare across different values of k. |
| **Silhouette Score** | Measures how similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Closer to 1 is better. Above 0.5 is good. |
Adjust k and watch inertia and the Silhouette Score together to find the best number of clusters.""")

            table1, table2, table3 = st.tabs(["Elbow Plot", "Cluster Scatter", "Silhouette Analysis"])

            with table1:
                # Compute inertia for k = 1 to 15 to draw the elbow curve
                k_range  = range(1, 16)
                inertias = []
                for k in k_range:
                    km = KMeans(
                        n_clusters=k,
                        init=model_params["init"],
                        n_init=model_params["n_init"],
                        max_iter=model_params["max_iter"],
                        random_state=model_params["random_state"])
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(list(k_range), inertias, marker="o", color="blue", lw=2)
                # Red dashed line marks the currently selected k
                ax.axvline(model_params["n_clusters"], color="red", linestyle="--", lw=1.5,
                           label=f"Current k = {model_params['n_clusters']}")
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Plot")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Look for the 'elbow' — the point where inertia starts decreasing more slowly. The red dashed line marks your current k. Try adjusting k to where the curve bends.")

            with table2:
                # Reduce to 2D with PCA so we can visualize clusters regardless of how many features were selected
                pca_2d  = PCA(n_components=2, random_state=int(random_state))
                X_2d    = pca_2d.fit_transform(X_scaled)
                var_exp = pca_2d.explained_variance_ratio_

                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                                     c=labels, cmap="tab10", alpha=0.7,
                                     edgecolors="white", s=50)
                # Overlay centroids projected into 2D space
                centers_2d = pca_2d.transform(model.cluster_centers_)
                ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                           c="black", marker="X", s=200, zorder=5, label="Centroids")
                ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
                ax.set_title("Cluster Scatter (PCA-reduced to 2D)")
                ax.legend()
                plt.colorbar(scatter, ax=ax, label="Cluster")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each color represents a cluster. X marks show the cluster centroids. Data is projected onto 2 principal components for visualization — the axis labels show how much variance each direction captures.")

            with table3:
                # Per-sample silhouette coefficients, grouped and sorted by cluster
                sil_vals   = silhouette_samples(X_scaled, labels)
                n_clusters = model_params["n_clusters"]
                colors     = plt.cm.tab10(np.linspace(0, 1, n_clusters))

                fig, ax = plt.subplots(figsize=(6, max(4, n_clusters * 0.8)))
                y_lower = 10
                for i in range(n_clusters):
                    cluster_sil = np.sort(sil_vals[labels == i])
                    size        = cluster_sil.shape[0]
                    y_upper     = y_lower + size
                    ax.fill_betweenx(np.arange(y_lower, y_upper),
                                     0, cluster_sil,
                                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
                    ax.text(-0.05, y_lower + 0.5 * size, str(i))
                    y_lower = y_upper + 10

                ax.axvline(sil_score, color="red", linestyle="--", lw=1.5,
                           label=f"Avg = {sil_score:.4f}")
                ax.set_xlabel("Silhouette coefficient")
                ax.set_ylabel("Cluster")
                ax.set_title("Silhouette Analysis by Cluster")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each colored band is a cluster. Wider bands mean more data points in that cluster. Bands that extend past the red average line indicate well-separated clusters. Thin or negative bands suggest that cluster may overlap with another.")

        ## PCA SECTION

        elif model_name == "PCA":

            # PCA requires numeric features only — filter and warn if non-numeric columns were selected
            numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_features) < 2:
                st.error("PCA requires at least 2 numeric feature columns. Please select more numeric columns.")
                st.stop()

            X_pca        = all_var[numeric_features].values
            X_pca_scaled = StandardScaler().fit_transform(X_pca)

            model         = PCA(**model_params)
            X_transformed = model.fit_transform(X_pca_scaled)

            # Compute metrics
            evr            = model.explained_variance_ratio_
            cumulative_var = np.cumsum(evr)
            n_components   = model_params["n_components"]

            # Present the metrics below the dataset preview
            st.subheader("Model Performance: PCA")
            col1, col2, col3 = st.columns(3)
            col1.metric("Components",         n_components)
            col2.metric("Variance Explained", f"{cumulative_var[-1]*100:.2f}%")
            col3.metric("PC1 Variance",       f"{evr[0]*100:.2f}%")

            # Explain what a good score looks like
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Components** | The number of principal components you chose to retain. Each one captures a different direction of variance in the data. |
| **Variance Explained** | Cumulative % of the original dataset's variance captured by all retained components. 80–90%+ is generally considered good. |
| **PC1 Variance** | % of variance captured by the first (strongest) principal component alone. A very high value means one direction dominates the data. |
Adjust the number of components and watch the cumulative variance change. Aim for the fewest components that still explain 80–90% of variance.""")

            table1, table2, table3 = st.tabs(["Scree Plot", "Cumulative Variance", "Component Loadings"])

            with table1:
                # Bar + line scree plot showing variance per component
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(1, n_components + 1), evr * 100,
                       color="blue", edgecolor="white", alpha=0.8)
                ax.plot(range(1, n_components + 1), evr * 100,
                        marker="o", color="red", lw=1.5)
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Variance Explained (%)")
                ax.set_title("Scree Plot")
                ax.set_xticks(range(1, n_components + 1))
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each bar shows how much variance a single principal component captures. Look for where the bars level off — components after that point add little new information.")

            with table2:
                # Cumulative variance curve with 80% and 90% reference lines
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, n_components + 1), cumulative_var * 100,
                        marker="o", color="blue", lw=2)
                ax.axhline(80, color="orange", linestyle="--", lw=1, label="80% threshold")
                ax.axhline(90, color="red",    linestyle="--", lw=1, label="90% threshold")
                ax.set_xlabel("Number of Components")
                ax.set_ylabel("Cumulative Variance Explained (%)")
                ax.set_title("Cumulative Explained Variance")
                ax.set_ylim(0, 105)
                ax.set_xticks(range(1, n_components + 1))
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Shows how much total variance is captured as you add more components. The orange and red lines mark the 80% and 90% thresholds — common targets for retaining enough information.")

            with table3:
                # Heatmap of how much each original feature contributes to each component
                loadings = pd.DataFrame(
                    model.components_.T,
                    index=numeric_features,
                    columns=[f"PC{i+1}" for i in range(n_components)])

                fig, ax = plt.subplots(figsize=(max(5, n_components * 0.9), max(4, len(numeric_features) * 0.45)))
                sns.heatmap(
                    loadings, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax)
                ax.set_title("Component Loadings")
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each cell shows how strongly a feature contributes to a principal component. Red = strong positive loading, blue = strong negative loading. Features with high absolute values drive that component.")

    # Neat error message instead of a long, ugly traceback
    except Exception as e:
        st.error(f"Model failed: {e}")
        st.exception(e)
