import os
import pandas as pd
import streamlit as st


# CSV dataset path
DATA_PATH = os.path.join("data", "student_placement.csv")


@st.cache_data
def load_data():
    """
    Load the student placement dataset from a CSV file.
    """

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        st.info(
            "Please create a 'data' folder and place "
            "'student_placement.csv' inside it."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_PATH)

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove completely empty rows
        df = df.dropna(how="all")

        return df

    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


def get_dataset_info(df):
    """
    Return basic information about the dataset.
    """

    if df.empty:
        return {
            "total_records": 0,
            "total_columns": 0,
            "missing_values": 0,
            "duplicate_rows": 0,
        }

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
