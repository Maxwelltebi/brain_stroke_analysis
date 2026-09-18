from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).parent
DATA_PATHS = (
    ROOT / "brain_stroke_dataset" / "healthcare-dataset-stroke-data.csv",
    ROOT / "healthcare-dataset-stroke-data.csv",
)
MODEL_PATH = ROOT / "neural_network_model.keras"
SCALER_PATH = ROOT / "standard_scaler.joblib"
LABEL_ENCODER_PATH = ROOT / "label_encoder.joblib"
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Marital Status",
    "BMI",
    "Occupation Type",
    "Residence Type",
    "Smoking Status",
    "Hypertension",
    "Heart Disease",
    "Average Glucose Level",
]
MODEL_CATEGORY_MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Marital Status": {"No": 1, "Yes": 0},
    "Occupation Type": {
        "children": 0,
        "Govt_job": 1,
        "Private": 2,
        "Self-employed": 3,
        "Never_worked": 4,
    },
    "Residence Type": {"Rural": 0, "Urban": 1},
    "Smoking Status": {
        "formerly smoked": 0,
        "never smoked": 1,
        "smokes": 2,
        "Unknown": 3,
    },
}
TEAL = "#00BFA5"
TEAL_DARK = "#00796B"
TEAL_LIGHT = "#E0F7FA"
CYAN_LIGHT = "#B2EBF2"


st.set_page_config(
    page_title="Brain Stroke Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    f"""
    <style>
    .stApp {{ background: {TEAL_LIGHT}; }}
    [data-testid="stSidebar"] {{ background: #ffffff; }}
    .block-container {{ max-width: 1180px; padding-top: 2rem; }}
    .section-title {{
        background: {TEAL_LIGHT};
        color: {TEAL_DARK};
        border: 1px solid {CYAN_LIGHT};
        border-radius: 15px 50px;
        padding: 0.65rem 1rem;
        margin: 1.2rem 0 1rem;
        font-weight: 700;
    }}
    .eyebrow {{
        color: {TEAL_DARK};
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    for data_path in DATA_PATHS:
        if data_path.exists():
            return pd.read_csv(data_path)
    searched_paths = ", ".join(str(path) for path in DATA_PATHS)
    raise FileNotFoundError(f"Dataset not found. Checked: {searched_paths}")


@st.cache_data
def prepare_data():
    frame = load_data().copy()
    frame = frame.drop_duplicates().dropna(subset=["bmi"])
    frame = frame[frame["gender"] != "Other"]
    frame["stroke_label"] = frame["stroke"].map({0: "No stroke", 1: "Stroke"})
    return frame


@st.cache_resource
def load_model_artifacts():
    missing = [str(path) for path in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing exported model files. Expected: "
            f"{MODEL_PATH.name}, {SCALER_PATH.name}, and {LABEL_ENCODER_PATH.name}."
        )

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to load the exported Keras model. Install it in the project runtime before running predictions."
        ) from exc

    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, scaler, label_encoder


def map_category(value, mapping):
    normalized = str(value).strip()
    key = mapping.get(normalized)
    if key is not None:
        return key
    for valid_value in mapping:
        if normalized.lower() == valid_value.lower():
            return mapping[valid_value]
    raise ValueError(f"Unrecognized category '{value}' for mapping {list(mapping.keys())}")


def build_feature_vector(
    age,
    gender,
    marital_status,
    bmi,
    occupation_type,
    residence_type,
    smoking_status,
    hypertension,
    heart_disease,
    avg_glucose_level,
):
    category_maps = {
        **MODEL_CATEGORY_MAPS,
        "Hypertension": {"No": 0, "Yes": 1},
        "Heart Disease": {"No": 0, "Yes": 1},
    }

    feature_vector = [
        float(age),
        map_category(gender, category_maps["Gender"]),
        map_category(marital_status, category_maps["Marital Status"]),
        float(bmi),
        map_category(occupation_type, category_maps["Occupation Type"]),
        map_category(residence_type, category_maps["Residence Type"]),
        map_category(smoking_status, category_maps["Smoking Status"]),
        map_category(hypertension, category_maps["Hypertension"]),
        map_category(heart_disease, category_maps["Heart Disease"]),
        float(avg_glucose_level),
    ]
    return np.asarray([feature_vector], dtype=float)


def select_decision_threshold(target, probabilities):
    thresholds = np.linspace(0.001, 0.15, 300)
    return max(
        thresholds,
        key=lambda threshold: f1_score(
            target,
            probabilities >= threshold,
            zero_division=0,
        ),
    )


@st.cache_data
def evaluate_model():
    frame = prepare_data()
    model_frame = pd.DataFrame(
        {
            "Age": frame["age"].astype(float),
            "Gender": frame["gender"].map(MODEL_CATEGORY_MAPS["Gender"]),
            "Marital Status": frame["ever_married"].map(MODEL_CATEGORY_MAPS["Marital Status"]),
            "BMI": frame["bmi"].astype(float),
            "Occupation Type": frame["work_type"].map(MODEL_CATEGORY_MAPS["Occupation Type"]),
            "Residence Type": frame["Residence_type"].map(MODEL_CATEGORY_MAPS["Residence Type"]),
            "Smoking Status": frame["smoking_status"].map(MODEL_CATEGORY_MAPS["Smoking Status"]),
            "Hypertension": frame["hypertension"].astype(float),
            "Heart Disease": frame["heart_disease"].astype(float),
            "Average Glucose Level": frame["avg_glucose_level"].astype(float),
        }
    )
    if model_frame.isna().any().any():
        raise ValueError("The validation data contains a category without a saved model mapping.")

    target = frame["stroke"].astype(int)
    training_features, validation_features, training_target, validation_target = train_test_split(
        model_frame,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model, scaler, _ = load_model_artifacts()
    training_probabilities = model.predict(scaler.transform(training_features[FEATURE_ORDER]), verbose=0).reshape(-1)
    scaled_features = scaler.transform(validation_features[FEATURE_ORDER])
    probabilities = model.predict(scaled_features, verbose=0).reshape(-1)
    threshold = select_decision_threshold(training_target, training_probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "validation_size": len(validation_target),
        "stroke_count": int(validation_target.sum()),
        "threshold": threshold,
        "accuracy": accuracy_score(validation_target, predictions),
        "precision": precision_score(validation_target, predictions, zero_division=0),
        "recall": recall_score(validation_target, predictions, zero_division=0),
        "f1": f1_score(validation_target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(validation_target, probabilities),
        "confusion_matrix": confusion_matrix(validation_target, predictions),
    }


def predict_stroke_probability(payload):
    model, scaler, label_encoder = load_model_artifacts()
    feature_vector = build_feature_vector(**payload)
    transformed = scaler.transform(feature_vector)
    prediction = model.predict(transformed, verbose=0)
    probability = float(prediction[0][0])
    threshold = evaluate_model()["threshold"]
    risk_label = "High risk" if probability >= threshold else "Low risk"
    return risk_label, probability, model, scaler, label_encoder


def section_title(number, title):
    st.markdown(f'<div class="section-title">{number}. {title}</div>', unsafe_allow_html=True)


def themed_figure(figure):
    figure.set_facecolor(TEAL_LIGHT)
    for axis in figure.axes:
        axis.set_facecolor(TEAL_LIGHT)
    st.pyplot(figure, width="stretch")
    plt.close(figure)


try:
    raw_data = load_data()
    data = prepare_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


with st.sidebar:
    st.markdown('<div class="eyebrow">Project presentation</div>', unsafe_allow_html=True)
    st.title("Brain Stroke Analysis")
    page = st.radio(
        "Navigate",
        ["Overview", "Dataset", "Exploration", "Method", "Test the model"],
    )
    st.divider()
    st.caption(f"{len(data):,} cleaned records available")


if page == "Overview":
    st.markdown('<div class="eyebrow">Healthcare analytics</div>', unsafe_allow_html=True)
    st.title("Brain Stroke Analysis & Prediction")
    st.write(
        "A guided walkthrough of the dataset, preparation decisions, exploratory findings, "
        "and machine-learning workflow behind the project."
    )
    st.info(
        "This project is educational and analytical. It is not a medical diagnosis or a substitute for professional care."
    )
    section_title("01", "Why stroke analysis matters")
    left, right = st.columns([1.25, 1])
    with left:
        st.write(
            "A stroke occurs when blood flow to part of the brain is interrupted or when a blood vessel ruptures. "
            "The project studies patient characteristics associated with recorded stroke outcomes."
        )
        st.write(
            "The presentation moves from raw data to a model-ready dataset, then ends with an interactive prediction workflow."
        )
    with right:
        st.metric("Original records", f"{len(raw_data):,}")
        st.metric("Cleaned records", f"{len(data):,}")
        st.metric("Recorded strokes", f"{int(data['stroke'].sum()):,}")

elif page == "Dataset":
    st.title("Dataset and preparation")
    section_title("02", "What the dataset contains")
    st.dataframe(data.head(10), width="stretch", hide_index=True)
    st.write(
        "The dataset includes demographic, lifestyle, clinical, and residential variables, with `stroke` as the target."
    )
    section_title("03", "Cleaning decisions")
    decisions = pd.DataFrame(
        {
            "Step": ["Remove duplicates", "Handle BMI missing values", "Remove ambiguous gender category", "Create readable labels"],
            "Reason": [
                "Avoid counting the same patient record more than once.",
                "BMI is used in the analysis and cannot be modeled as missing here.",
                "The exploratory analysis focuses on the two represented primary categories.",
                "Make charts and explanations easier to read.",
            ],
        }
    )
    st.dataframe(decisions, width="stretch", hide_index=True)
    st.write(f"Final analysis shape: **{data.shape[0]:,} rows x {data.shape[1]:,} columns**")

elif page == "Exploration":
    st.title("Exploratory analysis")
    section_title("04", "Stroke outcome balance")
    outcome = data["stroke_label"].value_counts().rename_axis("Outcome").reset_index(name="Patients")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=outcome, x="Outcome", y="Patients", palette=[TEAL, TEAL_DARK], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Patients")
    ax.set_title("Recorded stroke outcomes")
    themed_figure(fig)

    section_title("05", "Clinical and demographic patterns")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=data[data["stroke"] == 1], x="age", hue="gender", palette=[TEAL, TEAL_DARK], kde=True, ax=ax)
        ax.set_title("Age distribution among stroke records")
        themed_figure(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=data, x="stroke_label", y="avg_glucose_level", palette=[TEAL_LIGHT, TEAL], ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Average glucose level")
        ax.set_title("Glucose level by outcome")
        themed_figure(fig)

    section_title("06", "Correlation map")
    numeric = data.select_dtypes(include="number").drop(columns=["id"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric.corr(), cmap=[TEAL_LIGHT, CYAN_LIGHT, TEAL, TEAL_DARK], annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation among numeric variables")
    themed_figure(fig)

elif page == "Method":
    st.title("Methodical workflow")
    section_title("07", "From raw data to model input")
    steps = [
        ("Load", "Read the healthcare stroke dataset into a pandas DataFrame."),
        ("Clean", "Remove duplicates, handle missing BMI values, and remove the ambiguous gender category."),
        ("Transform", "Convert categorical values into a consistent representation for modeling."),
        ("Split", "Separate features and target, then divide the records into training and testing sets."),
        ("Scale", "Standardize numeric inputs so the neural network receives comparable feature ranges."),
        ("Evaluate", "Compare predictions with the held-out test set using classification metrics."),
    ]
    for number, (name, description) in enumerate(steps, start=1):
        with st.container(border=True):
            st.subheader(f"{number:02d} / {name}")
            st.write(description)
    st.success("The exported neural-network artifact and its matching preprocessing objects are connected.")

elif page == "Test the model":
    st.title("Test the model")
    section_title("08", "Patient parameters")
    st.write("Enter a hypothetical patient profile to score it with the exported neural-network model and saved preprocessing objects.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 1, 100, 50)
            bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
            glucose = st.number_input("Average glucose level", min_value=40.0, max_value=300.0, value=100.0, step=0.1)
            gender = st.selectbox("Gender", ["Female", "Male"])
            smoking = st.selectbox("Smoking status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])
        with right:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
            married = st.selectbox("Marital status", ["No", "Yes"])
            work_type = st.selectbox("Occupation type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence = st.selectbox("Residence type", ["Urban", "Rural"])
        submitted = st.form_submit_button("Run prediction", type="primary")

    if submitted:
        try:
            risk_label, probability, _, _, _ = predict_stroke_probability(
                {
                    "age": age,
                    "gender": gender,
                    "marital_status": married,
                    "bmi": bmi,
                    "occupation_type": work_type,
                    "residence_type": residence,
                    "smoking_status": smoking,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "avg_glucose_level": glucose,
                }
            )
            st.success(f"Prediction: {risk_label} ({probability:.2%} probability of stroke)")
            st.json(
                {
                    "age": age,
                    "gender": gender,
                    "marital_status": married,
                    "bmi": bmi,
                    "occupation_type": work_type,
                    "residence_type": residence,
                    "smoking_status": smoking,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "average_glucose_level": glucose,
                    "feature_order": FEATURE_ORDER,
                }
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ModuleNotFoundError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed because the model inputs did not match the saved preprocessing pipeline: {exc}")
