from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "brain_stroke_dataset" / "healthcare-dataset-stroke-data.csv"
TEAL = "#00BFA5"
TEAL_DARK = "#00796B"
TEAL_LIGHT = "#E0F7FA"
CYAN_LIGHT = "#B2EBF2"


st.set_page_config(
    page_title="Brain Stroke Analysis",
    page_icon=":drop_of_blood:",
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data
def prepare_data():
    frame = load_data().copy()
    frame = frame.drop_duplicates().dropna(subset=["bmi"])
    frame = frame[frame["gender"] != "Other"]
    frame["stroke_label"] = frame["stroke"].map({0: "No stroke", 1: "Stroke"})
    return frame


def section_title(number, title):
    st.markdown(f'<div class="section-title">{number}. {title}</div>', unsafe_allow_html=True)


def themed_figure(figure):
    figure.set_facecolor(TEAL_LIGHT)
    for axis in figure.axes:
        axis.set_facecolor(TEAL_LIGHT)
    st.pyplot(figure, use_container_width=True)
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
        ["Overview", "Dataset", "Exploration", "Method", "Model report", "Test the model"],
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
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)
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
    st.dataframe(decisions, use_container_width=True, hide_index=True)
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
    st.warning("The final neural-network artifact and its exact preprocessing objects will be connected after you export them from Colab.")

elif page == "Model report":
    st.title("Model performance report")
    section_title("08", "Evaluation summary")
    st.write(
        "This section is reserved for the exported neural-network results: accuracy, precision, recall, F1-score, "
        "confusion matrix, and training-history plots."
    )
    st.info("Add the exported model report or training-history artifact to populate this section.")
    st.code("model/brain_stroke_model.keras\nmodel/preprocessor.pkl\nreports/model_report.json", language="text")

elif page == "Test the model":
    st.title("Test the model")
    section_title("09", "Patient parameters")
    st.write("Enter a hypothetical patient profile. The prediction button will be connected once the exported model is added.")
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
            married = st.selectbox("Ever married", ["No", "Yes"])
            work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence = st.selectbox("Residence type", ["Urban", "Rural"])
        submitted = st.form_submit_button("Run prediction", type="primary")

    if submitted:
        st.info(
            "The input form is ready. The neural-network model and matching preprocessing pipeline are not in the project folder yet, "
            "so no clinical prediction has been generated."
        )
        st.json(
            {
                "age": age,
                "bmi": bmi,
                "average_glucose_level": glucose,
                "gender": gender,
                "smoking_status": smoking,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": married,
                "work_type": work_type,
                "residence_type": residence,
            }
        )
