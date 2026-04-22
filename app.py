import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Machine Learning Explorer")
st.markdown(
    "Upload a dataset (or pick a sample), choose a model, tune hyperparameters, "
    "and explore performance metrics — all in one place."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
SAMPLE_DATASETS = {
    "Iris (multiclass)": load_iris,
    "Breast Cancer (binary)": load_breast_cancer,
    "Wine (multiclass)": load_wine,
}

MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Random Forest": RandomForestClassifier,
}


def load_sklearn_dataset(loader_fn):
    data = loader_fn()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, "target"


def encode_dataframe(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Dataset")
    data_source = st.radio("Data source", ["Sample dataset", "Upload CSV"])

    df = None
    default_target = None

    if data_source == "Sample dataset":
        chosen = st.selectbox("Choose a sample", list(SAMPLE_DATASETS.keys()))
        df, default_target = load_sklearn_dataset(SAMPLE_DATASETS[chosen])
    else:
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("Awaiting CSV upload…")

    if df is not None:
        st.divider()
        st.header("2 · Columns")
        target_col = st.selectbox(
            "Target column",
            df.columns.tolist(),
            index=df.columns.tolist().index(default_target) if default_target else len(df.columns) - 1,
        )
        feature_cols = st.multiselect(
            "Feature columns",
            [c for c in df.columns if c != target_col],
            default=[c for c in df.columns if c != target_col],
        )

        st.divider()
        st.header("3 · Model")
        model_name = st.selectbox("Algorithm", list(MODELS.keys()))

        st.divider()
        st.header("4 · Hyperparameters")

        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        model_params = {}

        if model_name == "Logistic Regression":
            model_params["C"] = st.slider("C (regularization strength)", 0.01, 10.0, 1.0, 0.01)
            model_params["max_iter"] = st.slider("Max iterations", 100, 2000, 200, 100)
            solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
            model_params["solver"] = solver
            model_params["random_state"] = int(random_state)

        elif model_name == "Decision Tree":
            model_params["max_depth"] = st.slider("Max depth", 1, 20, 5)
            model_params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2)
            model_params["min_samples_leaf"] = st.slider("Min samples leaf", 1, 10, 1)
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            model_params["criterion"] = criterion
            model_params["random_state"] = int(random_state)

        elif model_name == "K-Nearest Neighbors":
            model_params["n_neighbors"] = st.slider("k (neighbors)", 1, 25, 5)
            model_params["weights"] = st.selectbox("Weights", ["uniform", "distance"])
            model_params["metric"] = st.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"])


        st.divider()
        train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

# ── Main Panel ────────────────────────────────────────────────────────────────
if df is None:
    st.info("👈 Select a dataset from the sidebar to get started.")
    st.stop()

# Data preview
with st.expander("📋 Dataset preview", expanded=False):
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Descriptive statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# ── Training ──────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training model…"):
        try:
            # Prepare data
            data_encoded = encode_dataframe(df)
            X = data_encoded[feature_cols].values
            y = data_encoded[target_col].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
            )

            # Scale features (especially important for LR and KNN)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Build and train
            ModelClass = MODELS[model_name]
            model = ModelClass(**model_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            classes = np.unique(y)
            n_classes = len(classes)
            is_binary = n_classes == 2
            avg = "binary" if is_binary else "weighted"

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

            # ROC AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                if is_binary:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            else:
                auc = None

            # ── Metric cards ──────────────────────────────────────────────────
            st.subheader("Model Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("AUC-ROC", f"{auc:.3f}" if auc is not None else "N/A")

            # ── Tabs ──────────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Confusion Matrix", "ROC Curve", "Feature Importances", "Classification Report"]
            )

            # Tab 1 — Confusion matrix
            with tab1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax
                )
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_title(f"Confusion Matrix — {model_name}")
                st.pyplot(fig)
                plt.close(fig)

            # Tab 2 — ROC Curve
            with tab2:
                if auc is not None and hasattr(model, "predict_proba"):
                    fig, ax = plt.subplots(figsize=(5, 4))
                    if is_binary:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#185FA5", lw=2)
                    else:
                        for i, cls in enumerate(classes):
                            fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_prob[:, i])
                            ax.plot(fpr, tpr, lw=1.5, label=f"Class {cls}")
                        ax.set_title(f"ROC Curve (one-vs-rest) — {model_name}")
                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title(f"ROC Curve — {model_name}")
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("ROC curve not available for this model.")

            # Tab 3 — Feature importances
            with tab3:
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feat_df = pd.DataFrame(
                        {"Feature": feature_cols, "Importance": importances}
                    ).sort_values("Importance", ascending=True)
                    fig, ax = plt.subplots(figsize=(5, max(3, len(feature_cols) * 0.3)))
                    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#1D9E75")
                    ax.set_xlabel("Importance")
                    ax.set_title(f"Feature Importances — {model_name}")
                    st.pyplot(fig)
                    plt.close(fig)
                elif hasattr(model, "coef_"):
                    coefs = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
                    feat_df = pd.DataFrame(
                        {"Feature": feature_cols, "Coefficient (abs)": coefs}
                    ).sort_values("Coefficient (abs)", ascending=True)
                    fig, ax = plt.subplots(figsize=(5, max(3, len(feature_cols) * 0.3)))
                    ax.barh(feat_df["Feature"], feat_df["Coefficient (abs)"], color="#534AB7")
                    ax.set_xlabel("|Coefficient|")
                    ax.set_title(f"Feature Coefficients — {model_name}")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Feature importances are not available for K-Nearest Neighbors.")

            # Tab 4 — Classification report
            with tab4:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)

            # Store result in session state so it persists without re-running
            st.session_state["last_result"] = {
                "model_name": model_name,
                "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
            }

        except Exception as e:
            st.error(f"Training failed: {e}")

elif "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.info(
        f"Last trained: **{r['model_name']}** — "
        f"Accuracy {r['acc']:.3f} · Precision {r['prec']:.3f} · "
        f"Recall {r['rec']:.3f} · AUC {r['auc']:.3f if r['auc'] else 'N/A'}"
    )
else:
    st.info("👈 Configure your model in the sidebar, then click **Train Model**.")
