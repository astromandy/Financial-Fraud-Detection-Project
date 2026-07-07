from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError as exc:  # pragma: no cover - import availability depends on environment
    shap = None
    SHAP_IMPORT_ERROR = str(exc)
else:
    SHAP_IMPORT_ERROR = None

try:
    from lime import lime_tabular
except ImportError as exc:  # pragma: no cover - import availability depends on environment
    lime_tabular = None
    LIME_IMPORT_ERROR = str(exc)
else:
    LIME_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model1.sav"
SAMPLE_DATA_PATH = BASE_DIR / "sample_data.csv"
SAMPLE_DATA_EXAMPLE_PATH = BASE_DIR / "sample_data.example.csv"
INSTANCE_DIR = BASE_DIR / "instance"
CACHE_DIR = INSTANCE_DIR / "cache"

for directory in (INSTANCE_DIR, CACHE_DIR, CACHE_DIR / "matplotlib", CACHE_DIR / "xdg"):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_FEATURES = [
    "card1",
    "card2",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "TransactionAmt",
    "P_emaildomain",
    "ProductCD",
    "DeviceType",
]

FEATURE_DESCRIPTIONS = {
    "card1": "Unique card identifier or first digits of card number (anonymized).",
    "card2": "Card issuer identifier or BIN number.",
    "card4": "Card network/category.",
    "card6": "Type of card used in the transaction.",
    "addr1": "Numeric representation of billing ZIP code.",
    "addr2": "Numeric code representing billing country.",
    "TransactionAmt": "Transaction amount in USD.",
    "P_emaildomain": "Purchaser email domain category.",
    "ProductCD": "Product category associated with the transaction.",
    "DeviceType": "Device used to place the transaction.",
}

CARD4_OPTIONS = {1: "Discover", 2: "Mastercard", 3: "American Express", 4: "Visa"}
CARD6_OPTIONS = {1: "Credit", 2: "Debit"}
EMAIL_OPTIONS = {0: "Gmail", 1: "Outlook", 2: "Mail.com", 3: "Other", 4: "Yahoo"}
PRODUCT_OPTIONS = {0: "C", 1: "H", 2: "R", 3: "S", 4: "W"}
DEVICE_OPTIONS = {1: "Mobile", 2: "Desktop"}


st.set_page_config(
    page_title="Financial Fraud Detection with XAI",
    page_icon="💰",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Keep final_model1.sav inside WebApp/."
        )

    try:
        with MODEL_PATH.open("rb") as model_file:
            return pickle.load(model_file)
    except ModuleNotFoundError as exc:
        if exc.name == "lightgbm":
            raise ModuleNotFoundError(
                "The serialized model depends on `lightgbm`. Install project dependencies with "
                "`pip install -r requirements.txt` before running the app."
            ) from exc
        raise


@st.cache_data
def load_sample_data():
    if SAMPLE_DATA_PATH.exists():
        sample_data = pd.read_csv(SAMPLE_DATA_PATH)
        source_label = "Loaded from WebApp/sample_data.csv"
        return sample_data, source_label, "real"

    if SAMPLE_DATA_EXAMPLE_PATH.exists():
        sample_data = pd.read_csv(SAMPLE_DATA_EXAMPLE_PATH)
        source_label = "Loaded bundled starter data from WebApp/sample_data.example.csv"
        return sample_data, source_label, "starter"

    rng = np.random.default_rng(42)
    fallback_data = pd.DataFrame(
        {
            "card1": rng.integers(1000, 20000, 200),
            "card2": rng.integers(100, 600, 200),
            "card4": rng.choice(list(CARD4_OPTIONS), 200, replace=True),
            "card6": rng.choice(list(CARD6_OPTIONS), 200, replace=True),
            "addr1": rng.integers(0, 500, 200),
            "addr2": rng.integers(0, 100, 200),
            "TransactionAmt": rng.lognormal(mean=6.5, sigma=0.8, size=200).round(2),
            "P_emaildomain": rng.choice(list(EMAIL_OPTIONS), 200, replace=True),
            "ProductCD": rng.choice(list(PRODUCT_OPTIONS), 200, replace=True),
            "DeviceType": rng.choice(list(DEVICE_OPTIONS), 200, replace=True),
            "isFraud": rng.choice([0, 1], 200, p=[0.965, 0.035]),
        }
    )
    source_label = (
        "Fallback synthetic data generated because WebApp/sample_data.csv is missing"
    )
    return fallback_data, source_label, "synthetic"


def get_reference_frame(sample_data: pd.DataFrame) -> pd.DataFrame:
    reference = sample_data.copy()
    for feature in MODEL_FEATURES:
        if feature not in reference.columns:
            reference[feature] = 0
    return reference[MODEL_FEATURES]


def get_default_values(reference_frame: pd.DataFrame) -> dict[str, float]:
    defaults = {}
    for column in MODEL_FEATURES:
        if column in reference_frame.columns and not reference_frame[column].dropna().empty:
            defaults[column] = float(reference_frame[column].median())
        else:
            defaults[column] = 0.0
    return defaults


def get_option_index(options: dict[int, str], default_value: float) -> int:
    option_keys = list(options)
    candidate = int(default_value)
    return option_keys.index(candidate) if candidate in options else 0


def predict_probability(model, input_frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)
        probability_array = np.asarray(probabilities)
        if probability_array.ndim == 2 and probability_array.shape[1] > 1:
            return float(probability_array[0][1])
        return float(probability_array.ravel()[0])

    prediction = model.predict(input_frame)
    return float(np.asarray(prediction).ravel()[0])


def normalize_shap_values(raw_values) -> np.ndarray:
    values = raw_values.values if hasattr(raw_values, "values") else raw_values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] > 1:
        return values[:, :, 1]
    return values


@st.cache_data
def get_shap_context(sample_data: pd.DataFrame):
    reference_frame = get_reference_frame(sample_data).iloc[: min(len(sample_data), 150)]

    if shap is None:
        return None, None, reference_frame, SHAP_IMPORT_ERROR

    try:
        explainer = shap.TreeExplainer(loaded_model)
        raw_values = explainer.shap_values(reference_frame)
        return explainer, normalize_shap_values(raw_values), reference_frame, None
    except Exception as exc:
        return None, None, reference_frame, str(exc)


def create_force_plot_plotly(
    shap_values: np.ndarray, feature_values: np.ndarray, feature_names: list[str]
):
    data = pd.DataFrame(
        {
            "Feature": feature_names,
            "Value": feature_values,
            "Contribution": shap_values,
        }
    )
    data["AbsContribution"] = data["Contribution"].abs()
    data = data.sort_values("AbsContribution", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=data["Feature"],
            x=data["Contribution"],
            orientation="h",
            marker_color=["#d94f45" if x < 0 else "#2a7de1" for x in data["Contribution"]],
            text=[f"{name}: {value}" for name, value in zip(data["Feature"], data["Value"])],
            hoverinfo="text+x",
        )
    )
    fig.update_layout(
        title="SHAP Feature Contribution",
        xaxis_title="Contribution to Fraud Prediction",
        height=420,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def render_environment_notes(model_error: str | None, sample_data_mode: str, sample_source: str):
    if model_error:
        st.error(model_error)
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model path: `{MODEL_PATH}`")
    st.sidebar.caption(sample_source)

    if sample_data_mode == "starter":
        st.sidebar.info(
            "Using the bundled starter dataset for XAI examples. Add `WebApp/sample_data.csv` if you "
            "want a richer local reference dataset."
        )

    if sample_data_mode == "synthetic":
        st.sidebar.warning(
            "Real sample data was not found. The XAI dashboard is running on deterministic synthetic "
            "data, so treat those explanations as demo-only until you add `WebApp/sample_data.csv`."
        )

    if SHAP_IMPORT_ERROR:
        st.sidebar.info(
            "SHAP is not installed in this environment. Prediction still works, but SHAP charts stay hidden."
        )

    if LIME_IMPORT_ERROR:
        st.sidebar.info(
            "LIME is not installed in this environment. The LIME tab stays unavailable until you install it."
        )


try:
    loaded_model = load_model()
    model_error_message = None
except Exception as exc:  # pragma: no cover - depends on local assets
    loaded_model = None
    model_error_message = (
        "Could not load the trained model. "
        f"Details: {exc}"
    )

sample_data, sample_source, sample_data_mode = load_sample_data()
reference_frame = get_reference_frame(sample_data)
default_values = get_default_values(reference_frame)
important_features = [feature for feature in MODEL_FEATURES if feature in reference_frame.columns]


st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Prediction", "XAI Dashboard"])
render_environment_notes(model_error_message, sample_data_mode, sample_source)

if app_mode == "Prediction":
    st.title("Financial Transaction Fraud Prediction")
    st.write("Enter transaction details to estimate the probability of fraud.")

    with st.expander("Feature Descriptions Reference"):
        st.markdown(
            """
            | Feature | Description |
            | ------- | ----------- |
            | Card 1 | Unique card identifier or first digits of card number (anonymized) |
            | Card 2 | Card issuer identifier or BIN number |
            | Card 4 | Card network/category |
            | Card 6 | Card type |
            | Address 1 | Numeric representation of billing ZIP code |
            | Address 2 | Numeric code representing billing country |
            | Transaction Amount | Transaction amount in USD |
            | P Email Domain | Purchaser email domain category |
            | Product Code | Product category |
            | Device Type | Mobile or desktop |
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        card1 = st.number_input(
            "Card 1",
            min_value=1000,
            max_value=20000,
            value=int(default_values["card1"]),
            help=FEATURE_DESCRIPTIONS["card1"],
        )
        card2 = st.number_input(
            "Card 2",
            min_value=100,
            max_value=600,
            value=int(default_values["card2"]),
            help=FEATURE_DESCRIPTIONS["card2"],
        )
        card4 = st.selectbox(
            "Card 4 (Category)",
            list(CARD4_OPTIONS),
            index=get_option_index(CARD4_OPTIONS, default_values["card4"]),
            format_func=lambda value: CARD4_OPTIONS[value],
            help=FEATURE_DESCRIPTIONS["card4"],
        )
        card6 = st.selectbox(
            "Card 6 (Type)",
            list(CARD6_OPTIONS),
            index=get_option_index(CARD6_OPTIONS, default_values["card6"]),
            format_func=lambda value: CARD6_OPTIONS[value],
            help=FEATURE_DESCRIPTIONS["card6"],
        )
        addr1 = st.slider(
            "Address 1 (ZIP Code)",
            0,
            500,
            int(default_values["addr1"]),
            help=FEATURE_DESCRIPTIONS["addr1"],
        )

    with col2:
        addr2 = st.slider(
            "Address 2 (Country Code)",
            0,
            100,
            int(default_values["addr2"]),
            help=FEATURE_DESCRIPTIONS["addr2"],
        )
        transaction_amt = st.number_input(
            "Transaction Amount (USD)",
            min_value=0.0,
            max_value=50000.0,
            value=float(default_values["TransactionAmt"]),
            step=10.0,
            help=FEATURE_DESCRIPTIONS["TransactionAmt"],
        )
        email_domain = st.selectbox(
            "P Email Domain",
            list(EMAIL_OPTIONS),
            index=get_option_index(EMAIL_OPTIONS, default_values["P_emaildomain"]),
            format_func=lambda value: EMAIL_OPTIONS[value],
            help=FEATURE_DESCRIPTIONS["P_emaildomain"],
        )
        product_cd = st.selectbox(
            "Product Code",
            list(PRODUCT_OPTIONS),
            index=get_option_index(PRODUCT_OPTIONS, default_values["ProductCD"]),
            format_func=lambda value: PRODUCT_OPTIONS[value],
            help=FEATURE_DESCRIPTIONS["ProductCD"],
        )
        device_type = st.selectbox(
            "Device Type",
            list(DEVICE_OPTIONS),
            index=get_option_index(DEVICE_OPTIONS, default_values["DeviceType"]),
            format_func=lambda value: DEVICE_OPTIONS[value],
            help=FEATURE_DESCRIPTIONS["DeviceType"],
        )

    input_frame = pd.DataFrame(
        [
            {
                "card1": card1,
                "card2": card2,
                "card4": card4,
                "card6": card6,
                "addr1": addr1,
                "addr2": addr2,
                "TransactionAmt": transaction_amt,
                "P_emaildomain": email_domain,
                "ProductCD": product_cd,
                "DeviceType": device_type,
            }
        ]
    )[MODEL_FEATURES]

    if st.button("Predict Fraud Risk"):
        try:
            prediction_probability = predict_probability(loaded_model, input_frame)
            st.subheader("Prediction Result")
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.metric(label="Fraud Probability", value=f"{prediction_probability:.2%}")
            with result_col2:
                if prediction_probability >= 0.5:
                    st.error("⚠️ High likelihood of fraud detected.")
                else:
                    st.success("✅ Transaction appears legitimate.")

            st.subheader("Explanation of Prediction")
            explainer, _, _, shap_error = get_shap_context(sample_data)
            if explainer is not None:
                raw_input_shap = explainer.shap_values(input_frame)
                shap_for_display = normalize_shap_values(raw_input_shap)[0]
                st.write("Feature contribution to this prediction:")
                st.plotly_chart(
                    create_force_plot_plotly(
                        shap_for_display,
                        input_frame.iloc[0].values,
                        input_frame.columns.tolist(),
                    ),
                    use_container_width=True,
                )

                top_features = pd.DataFrame(
                    {
                        "Feature": input_frame.columns,
                        "Importance": np.abs(shap_for_display),
                    }
                ).sort_values("Importance", ascending=False).head(5)

                fig = px.bar(
                    top_features,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 5 Influential Features",
                    color="Importance",
                    color_continuous_scale=px.colors.sequential.Bluered,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "SHAP explanation is unavailable in this environment."
                    + (f" Details: {shap_error}" if shap_error else "")
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.error("Review the inputs and confirm the runtime dependencies are installed.")

else:
    st.title("XAI Dashboard for Financial Fraud Detection")
    st.write("Explore global and local explanations for the fraud detection model.")

    tab1, tab2, tab3 = st.tabs(
        ["Feature Importance", "SHAP Analysis", "LIME Explanations"]
    )

    with tab1:
        st.header("Feature Importance")
        st.write("Understand which features matter most to the model.")

        if hasattr(loaded_model, "feature_importances_"):
            try:
                importances = loaded_model.feature_importances_
                feature_names = reference_frame.columns.tolist()

                if len(importances) != len(feature_names):
                    feature_names = [f"feature_{index}" for index in range(len(importances))]
                    st.warning(
                        "The model feature count does not match the reference dataset. "
                        "Generic feature names are shown below."
                    )

                feature_importance = pd.DataFrame(
                    {"Feature": feature_names, "Importance": importances}
                ).sort_values("Importance", ascending=False)

                fig = px.bar(
                    feature_importance.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance",
                    color="Importance",
                    color_continuous_scale=px.colors.sequential.Viridis,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(feature_importance, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not render feature importance: {exc}")
        elif "isFraud" in sample_data.columns:
            st.info("Direct feature importance is unavailable. Using permutation importance instead.")
            try:
                permutation = permutation_importance(
                    loaded_model,
                    reference_frame,
                    sample_data["isFraud"],
                    n_repeats=5,
                    random_state=42,
                )
                permutation_df = pd.DataFrame(
                    {
                        "Feature": reference_frame.columns,
                        "Importance": permutation.importances_mean,
                    }
                ).sort_values("Importance", ascending=False)
                fig = px.bar(
                    permutation_df.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Permutation Importance",
                    color="Importance",
                    color_continuous_scale=px.colors.sequential.Viridis,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not calculate permutation importance: {exc}")
        else:
            st.warning(
                "Feature importance is unavailable because the reference dataset has no `isFraud` labels."
            )

    with tab2:
        st.header("SHAP Analysis")
        st.write(
            "SHAP values show how each feature pushes predictions toward higher or lower fraud risk."
        )

        explainer, shap_values, shap_reference, shap_error = get_shap_context(sample_data)
        if explainer is None or shap_values is None:
            st.warning(
                "SHAP visualizations are unavailable right now."
                + (f" Details: {shap_error}" if shap_error else "")
            )
        else:
            summary_plot_type = st.radio("Summary plot type", ["Bar", "Beeswarm", "Violin"])
            try:
                fig, _ = plt.subplots(figsize=(10, 8))
                if summary_plot_type == "Bar":
                    shap.summary_plot(shap_values, shap_reference, plot_type="bar", show=False)
                elif summary_plot_type == "Beeswarm":
                    shap.summary_plot(shap_values, shap_reference, show=False)
                else:
                    shap.summary_plot(shap_values, shap_reference, plot_type="violin", show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Could not create the SHAP summary plot: {exc}")

            selected_feature = st.selectbox("Feature for dependence plot", important_features)
            feature_index = shap_reference.columns.tolist().index(selected_feature)
            feature_values = shap_reference[selected_feature].to_numpy()
            shap_feature_values = shap_values[:, feature_index]

            dependence_fig = px.scatter(
                x=feature_values,
                y=shap_feature_values,
                title=f"SHAP Dependence Plot for {selected_feature}",
                labels={"x": selected_feature, "y": "SHAP value"},
                opacity=0.7,
                color=np.abs(shap_feature_values),
                color_continuous_scale="Viridis",
            )
            dependence_fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(dependence_fig, use_container_width=True)

    with tab3:
        st.header("LIME Explanations")
        st.write("Inspect a single sample and see which features pushed the model decision.")

        if lime_tabular is None:
            st.warning(
                "LIME is unavailable in this environment."
                + (f" Details: {LIME_IMPORT_ERROR}" if LIME_IMPORT_ERROR else "")
            )
        elif not hasattr(loaded_model, "predict_proba"):
            st.warning("LIME requires a model with `predict_proba`, which is unavailable here.")
        else:
            selected_index = st.slider(
                "Select sample to explain",
                min_value=0,
                max_value=max(len(reference_frame) - 1, 0),
                value=0,
            )

            selected_sample = reference_frame.iloc[[selected_index]]
            st.write("Selected sample features:")
            st.dataframe(selected_sample, use_container_width=True)

            probability = predict_probability(loaded_model, selected_sample)
            st.write(f"Model prediction: `{probability:.2%}` probability of fraud")

            categorical_features = [
                reference_frame.columns.get_loc(column)
                for column in ["card4", "card6", "P_emaildomain", "ProductCD", "DeviceType"]
                if column in reference_frame.columns
            ]

            try:
                lime_explainer = lime_tabular.LimeTabularExplainer(
                    training_data=reference_frame.to_numpy(),
                    feature_names=reference_frame.columns.tolist(),
                    class_names=["Legitimate", "Fraud"],
                    categorical_features=categorical_features,
                    mode="classification",
                )

                explanation = lime_explainer.explain_instance(
                    selected_sample.iloc[0].to_numpy(),
                    loaded_model.predict_proba,
                    num_features=min(10, len(reference_frame.columns)),
                )

                explanation_items = explanation.as_list()
                feature_labels, contribution_values = zip(*explanation_items)

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=contribution_values,
                        y=feature_labels,
                        orientation="h",
                        marker_color=[
                            "#d94f45" if value < 0 else "#2f9e44"
                            for value in contribution_values
                        ],
                    )
                )
                fig.update_layout(
                    title="LIME Explanation",
                    xaxis_title="Contribution to Prediction",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not generate the LIME explanation: {exc}")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    This Streamlit app predicts fraud probability for financial transactions and adds
    explainability with feature importance, SHAP, and LIME when those dependencies are available.
    """
)
