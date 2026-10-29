import os

import joblib
import pandas as pd
import streamlit as st

from chart import show_charts
from data_loader import load_data
from style import load_css, show_header, show_footer


# -----------------------------------------------------
# PAGE CONFIGURATION AND STYLING
# -----------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
)

load_css()
show_header()


# -----------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------
# @st.cache_data
def get_dataset():
    return load_data()


try:
    df = get_dataset()
except Exception as exc:
    st.error(f"Unable to load the student dataset: {exc}")
    st.info(
        "Check data_loader.py and make sure the dataset file "
        "exists at the path configured there."
    )
    show_footer()
    st.stop()


# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Placement Analytics",
        "Predict Placement",
        "Dataset Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "This application demonstrates student placement "
    "prediction using machine learning and academic "
    "performance data."
)


# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------
if page == "Home":
    st.subheader("🏠 Project Overview")
    st.write(
        "The Student Placement Prediction System uses machine "
        "learning to estimate placement outcomes from academic "
        "and employability indicators."
    )

    total_students = len(df)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Student Records", f"{total_students:,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()
    st.subheader("📋 Dataset Preview")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Use the navigation menu to explore placement analytics, "
        "make a prediction, or search the dataset."
    )


# -----------------------------------------------------
# PLACEMENT ANALYTICS PAGE
# -----------------------------------------------------
elif page == "Placement Analytics":
    st.title("📊 Student Placement Analytics")

    try:
        show_charts(df)
    except Exception as exc:
        st.error(f"Unable to display analytics charts: {exc}")


# -----------------------------------------------------
# PREDICT PLACEMENT PAGE
# -----------------------------------------------------
elif page == "Predict Placement":
    st.title("🔮 Predict Student Placement")
    st.write(
        "Enter the student's details to estimate the placement outcome."
    )

    model_path = os.path.join("models", "placement_model.pkl")

    if not os.path.exists(model_path):
        st.error(
            "Trained model not found. Train the model first and save it "
            "to models/placement_model.pkl."
        )
    else:
        try:
            saved_object = joblib.load(model_path)

            # Support both a model saved directly and the model bundle
            # produced by train_model_fixed.py.
            if isinstance(saved_object, dict) and "model" in saved_object:
                model = saved_object["model"]
                feature_names = saved_object.get("feature_columns", [])
                saved_classes = saved_object.get("classes", [])
            else:
                model = saved_object
                saved_classes = []
                feature_names = []

            # If feature names were not included in the saved bundle,
            # try to read them from the fitted estimator/pipeline.
            if not feature_names and hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            elif not feature_names and hasattr(model, "named_steps"):
                preprocessor = model.named_steps.get("preprocessor")
                if preprocessor is not None and hasattr(
                    preprocessor, "feature_names_in_"
                ):
                    feature_names = list(preprocessor.feature_names_in_)

            if not feature_names:
                st.error(
                    "The model's input columns could not be identified. "
                    "Please save feature_columns in the model bundle or "
                    "train the model using a pandas DataFrame."
                )
            else:
                numeric_columns = df.select_dtypes(
                    include=["number"]
                ).columns.tolist()

                categorical_columns = [
                    feature
                    for feature in feature_names
                    if feature in df.columns
                    and feature not in numeric_columns
                ]

                with st.form("placement_prediction_form"):
                    st.subheader("Student Information")
                    form_values = {}
                    col1, col2 = st.columns(2)

                    for index, feature in enumerate(feature_names):
                        container = col1 if index % 2 == 0 else col2

                        with container:
                            if (
                                feature in numeric_columns
                                and feature in df.columns
                            ):
                                values = pd.to_numeric(
                                    df[feature], errors="coerce"
                                ).dropna()

                                if values.empty:
                                    form_values[feature] = st.number_input(
                                        feature, value=0.0
                                    )
                                else:
                                    minimum = float(values.min())
                                    maximum = float(values.max())
                                    median = float(values.median())

                                    if minimum == maximum:
                                        form_values[feature] = st.number_input(
                                            feature, value=median
                                        )
                                    else:
                                        form_values[feature] = st.number_input(
                                            feature,
                                            min_value=minimum,
                                            max_value=maximum,
                                            value=min(
                                                max(median, minimum), maximum
                                            ),
                                        )

                            elif (
                                feature in categorical_columns
                                and feature in df.columns
                            ):
                                options = sorted(
                                    df[feature]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )

                                if options:
                                    form_values[feature] = st.selectbox(
                                        feature, options=options
                                    )
                                else:
                                    form_values[feature] = st.text_input(
                                        feature
                                    )
                            else:
                                form_values[feature] = st.text_input(
                                    feature
                                )

                    submitted = st.form_submit_button(
                        "🔍 Predict Placement",
                        use_container_width=True,
                    )

                if submitted:
                    input_df = pd.DataFrame(
                        [form_values], columns=feature_names
                    )

                    try:
                        prediction = model.predict(input_df)[0]
                        st.divider()
                        st.subheader("Prediction Result")
                        st.success(
                            f"Predicted Placement Outcome: {prediction}"
                        )

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(input_df)[0]
                            classes = getattr(
                                model, "classes_", saved_classes
                            )

                            if len(classes) == len(probabilities):
                                probability_df = pd.DataFrame(
                                    {
                                        "Outcome": classes,
                                        "Probability (%)": (
                                            probabilities * 100
                                        ).round(2),
                                    }
                                )
                                st.markdown("#### Prediction Probabilities")
                                st.dataframe(
                                    probability_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                    except Exception as exc:
                        st.error(
                            "Prediction failed. Make sure the form inputs "
                            "match the model's training features. "
                            f"Details: {exc}"
                        )

        except Exception as exc:
            st.error(f"Unable to load the trained model: {exc}")


# -----------------------------------------------------
# DATASET EXPLORER PAGE
# -----------------------------------------------------
elif page == "Dataset Explorer":
    st.title("🗂️ Student Dataset Explorer")
    st.write("Search, inspect, and download records from the dataset.")

    search_text = st.text_input(
        "Search dataset",
        placeholder="Enter a student attribute or value",
    )

    filtered_df = df.copy()

    if search_text:
        mask = filtered_df.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)
        filtered_df = filtered_df[mask]

    st.write(f"Showing {len(filtered_df):,} of {len(df):,} records")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name="student_placement_data.csv",
        mime="text/csv",
    )


# -----------------------------------------------------
# FOOTER (keep at the end of the application)
# -----------------------------------------------------
st.divider()
show_footer()
