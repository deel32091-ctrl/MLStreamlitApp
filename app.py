import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
)

## Page setup
st.set_page_config(
    page_title="Supervised Machine Learning Tool",
    layout="wide")

st.title("Supervised Machine Learning Tool")
st.markdown(
    "Upload a dataset, experiment with hyperparameters, and observe how these affect model training and performance.")

def is_continuous(series):
    """Numeric column with more than 10 unique values → treat as continuous."""
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10

## Sidebar
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type="csv")

    df = None
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        st.info("Upload a CSV file")

    if df is not None:
        st.divider()
        st.header("Choose Variables")
        target_col = st.selectbox(
            "Variable to estimate",
            df.columns.tolist(),
            index=len(df.columns) - 1,
        )
        feature_cols = st.multiselect(
            "Features",
            [c for c in df.columns if c != target_col],
            default=[c for c in df.columns if c != target_col],
        )

        st.divider()
        st.header("Choose a Model")
        model_name = st.selectbox(
            "Algorithm",
            ["Linear Regression", "Logistic Regression"],
            help=(
                "Linear Regression for continuous numeric targets  \n"
                "Logistic Regression for binary targets"
            ),
        )

        st.divider()
        st.header("Tune Hyperparameters")

        random_state = st.number_input("Random seed", value=100, step=1)
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)

        model_params: dict = {}

        if model_name == "Linear Regression":
            model_params["alpha"] = st.slider(
                "Regularization parameter (α)",
                0.0, 10.0, 1.0, 0.1,
                help="Higher α → stronger regularization, shrinks coefficients toward zero. α = 0 is equivalent to plain OLS.",
            )

        elif model_name == "Logistic Regression":
            model_params["C"] = st.slider(
                "C (inverse regularization strength)",
                0.01, 10.0, 1.0, 0.01,
                help="Smaller C → stronger regularization.",
            )
            model_params["penalty"] = st.selectbox(
                "Penalty",
                ["l2", "l1"],
                help="L2 shrinks all coefficients toward zero; L1 can zero out coefficients entirely (built-in feature selection).",
            )
            model_params["solver"] = "liblinear"
            model_params["max_iter"] = 1000
            model_params["random_state"] = int(random_state)

        st.divider()
        train_btn = st.button("Train Model", use_container_width=True, type="primary")

## Main panel
if df is None:
    st.info("Upload a dataset to perform analysis.")
    st.stop()

with st.expander("Quick view of Dataset", expanded=True):
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(11), use_container_width=True)
    st.write("**Descriptive statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

## Training
if train_btn:
    with st.spinner("Training model"):
        try:
            # ── Model / target compatibility check ────────────────────────────
            target_is_continuous = is_continuous(df[target_col])

            if model_name == "Linear Regression" and not target_is_continuous:
                st.error(
                    f"**'{target_col}'** does not look like a continuous numeric target. "
                    "Linear Regression requires a continuous numeric target. "
                    "Please select a different target column or switch to **Logistic Regression**."
                )
                st.stop()

            if model_name == "Logistic Regression" and target_is_continuous:
                st.error(
                    f"**'{target_col}'** does not look like a binary target. "
                    "Logistic Regression requires a binary categorical target. "
                    "Please select a different target column or switch to **Linear Regression**."
                )
                st.stop()

            # ── Data preparation ──────────────────────────────────────────────
            from sklearn.preprocessing import OrdinalEncoder
            working = df[feature_cols + [target_col]].copy()

            # For logistic regression, save original target labels before encoding
            original_target_labels = None
            if model_name == "Logistic Regression":
                le = LabelEncoder()
                original_target_labels = le.classes_ if hasattr(le, 'classes_') else None
                # Fit on the full target column to capture all label names
                le.fit(working[target_col].astype(str))
                original_target_labels = le.classes_
                working[target_col] = le.transform(working[target_col].astype(str))

            # Encode any remaining text feature columns
            for col in working[feature_cols].select_dtypes(include=["object", "category"]).columns:
                working[col] = OrdinalEncoder().fit_transform(working[[col]])

            before = len(working)
            working = working.dropna()
            dropped = before - len(working)
            if dropped > 0:
                st.warning(f"⚠️ {dropped} row(s) with missing values were dropped before training.")

            X = working[feature_cols].values
            y = working[target_col].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=float(test_size),
                random_state=int(random_state),
                stratify=(y if model_name == "Logistic Regression" else None),
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # ── LINEAR REGRESSION ─────────────────────────────────────────────
            if model_name == "Linear Regression":
                model = Ridge(**model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2   = r2_score(y_test, y_pred)
                mse  = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae  = mean_absolute_error(y_test, y_pred)

                st.subheader("Model Performance — Linear Regression")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R²", f"{r2:.4f}")
                c2.metric("MSE", f"{mse:.4f}")
                c3.metric("RMSE", f"{rmse:.4f}")
                c4.metric("MAE", f"{mae:.4f}")

                tab1, tab2, tab3 = st.tabs(
                    ["Predicted Values vs Actual Values", "Residuals", "Feature Coefficients"]
                )

                with tab1:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(y_test, y_pred, alpha=0.6, color="#185FA5", edgecolors="white", s=50)
                    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
                    ax.set_xlabel("Actual values")
                    ax.set_ylabel("Predicted values")
                    ax.set_title("Predicted vs Actual")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                with tab2:
                    residuals = y_test - y_pred
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#E05A3A", edgecolors="white", s=50)
                    axes[0].axhline(0, color="black", lw=1.5, linestyle="--")
                    axes[0].set_xlabel("Predicted values")
                    axes[0].set_ylabel("Residuals")
                    axes[0].set_title("Residuals vs Fitted")
                    axes[1].hist(residuals, bins=20, color="#534AB7", edgecolor="white")
                    axes[1].set_xlabel("Residual")
                    axes[1].set_ylabel("Frequency")
                    axes[1].set_title("Residual Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with tab3:
                    coef_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Coefficient": model.coef_
                    }).sort_values("Coefficient", key=abs, ascending=True)
                    fig, ax = plt.subplots(figsize=(5, max(3, len(feature_cols) * 0.35)))
                    colors = ["#E05A3A" if c < 0 else "#1D9E75" for c in coef_df["Coefficient"]]
                    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
                    ax.axvline(0, color="black", lw=1)
                    ax.set_xlabel("Coefficient value")
                    ax.set_title("Feature Coefficients\n(green = positive effect, red = negative)")
                    st.pyplot(fig)
                    plt.close(fig)

                st.session_state["last_result"] = {
                    "model_name": model_name,
                    "r2": r2, "mse": mse, "rmse": rmse, "mae": mae,
                }

            # ── LOGISTIC REGRESSION ───────────────────────────────────────────
            elif model_name == "Logistic Regression":
                model = LogisticRegression(**model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Use original label names for display
                class_indices = np.unique(y).astype(int)
                display_labels = original_target_labels[class_indices] if original_target_labels is not None else class_indices

                n_classes = len(class_indices)
                is_binary = n_classes == 2
                avg = "binary" if is_binary else "weighted"

                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
                rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
                f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

                y_prob = model.predict_proba(X_test)
                if is_binary:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

                st.subheader("Model Performance — Logistic Regression")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy",  f"{acc:.4f}")
                c2.metric("Precision", f"{prec:.4f}")
                c3.metric("Recall",    f"{rec:.4f}")
                c4.metric("F1 Score",  f"{f1:.4f}")
                c5.metric("AUC-ROC",   f"{auc:.4f}")

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Confusion Matrix", "ROC Curve", "Feature Coefficients", "Classification Report"]
                )

                with tab1:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=display_labels, yticklabels=display_labels, ax=ax
                    )
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("True label")
                    ax.set_title("Confusion Matrix — Logistic Regression")
                    st.pyplot(fig)
                    plt.close(fig)

                with tab2:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    if is_binary:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#185FA5", lw=2)
                    else:
                        for i, (idx, lbl) in enumerate(zip(class_indices, display_labels)):
                            fpr, tpr, _ = roc_curve((y_test == idx).astype(int), y_prob[:, i])
                            ax.plot(fpr, tpr, lw=1.5, label=str(lbl))
                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curve — Logistic Regression")
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    plt.close(fig)

                with tab3:
                    coefs = (
                        np.abs(model.coef_).mean(axis=0)
                        if model.coef_.ndim > 1
                        else model.coef_[0]
                    )
                    coef_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Coefficient": coefs,
                    }).sort_values("Coefficient", key=abs, ascending=True)
                    fig, ax = plt.subplots(figsize=(5, max(3, len(feature_cols) * 0.35)))
                    colors = ["#E05A3A" if c < 0 else "#1D9E75" for c in coef_df["Coefficient"]]
                    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
                    ax.axvline(0, color="black", lw=1)
                    ax.set_xlabel("Coefficient value")
                    ax.set_title("Feature Coefficients\n(green = positive class, red = negative class)")
                    st.pyplot(fig)
                    plt.close(fig)

                with tab4:
                    report = classification_report(
                        y_test, y_pred,
                        target_names=display_labels,
                        output_dict=True, zero_division=0
                    )
                    st.dataframe(
                        pd.DataFrame(report).transpose().style.format(precision=3),
                        use_container_width=True,
                    )

                st.session_state["last_result"] = {
                    "model_name": model_name,
                    "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
                }

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)

else:
    st.info("After uploading the dataset, configure the model and click Train Model.")
