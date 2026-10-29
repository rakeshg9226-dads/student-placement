import os
import re
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_placement.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "placement_model.pkl")

TARGET_COLUMN = "Placement Status"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and validate CSV dataset
# --------------------------------------------------
def load_dataset():
    if not os.path.isfile(DATA_PATH):
        # Also allow the CSV to be placed next to train_model.py.
        alternate_path = os.path.join(BASE_DIR, "student_placement.csv")
        if os.path.isfile(alternate_path):
            dataset_path = alternate_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at:\\n{DATA_PATH}\\n\\n"
                "Place student_placement.csv inside the project's data folder "
                "or next to train_model.py."
            )
    else:
        dataset_path = DATA_PATH

    # sep=None + engine='python' detects comma-, tab-, or semicolon-separated files.
    df = pd.read_csv(dataset_path, encoding="utf-8-sig", sep=None, engine="python")

    # Clean whitespace and invisible BOM characters from headers.
    df.columns = [
        str(col).replace("\\ufeff", "").strip()
        for col in df.columns
    ]

    print("Columns found in CSV:", df.columns.tolist())

    def normalize_column_name(name):
        # Ignore case, spaces, underscores, punctuation, and hidden separators.
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    target_key = normalize_column_name(TARGET_COLUMN)
    matches = [
        col for col in df.columns
        if normalize_column_name(col) == target_key
    ]

    if not matches:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Detected columns: {df.columns.tolist()}. "
            "Check that the CSV contains a placement-result column."
        )

    actual_target = matches[0]
    if actual_target != TARGET_COLUMN:
        df.rename(columns={actual_target: TARGET_COLUMN}, inplace=True)

    df = df.dropna(how="all").drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df = df[df[TARGET_COLUMN] != ""]

    if df.empty:
        raise ValueError("The CSV has no usable student records.")

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            f"'{TARGET_COLUMN}' must contain at least two classes, "
            "for example 'Placed' and 'Not Placed'."
        )

    return df


# --------------------------------------------------
# 3. Prepare model features
# --------------------------------------------------
def prepare_features(df):
    y = df[TARGET_COLUMN].astype(str).str.strip()
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    # IDs identify students but should not be used as predictive features.
    id_columns = [
        col for col in X.columns
        if col.strip().lower().replace("_", " ") in {
            "student id", "roll number", "student number", "registration number"
        }
    ]
    X = X.drop(columns=id_columns)

    if X.shape[1] == 0:
        raise ValueError("No usable input feature columns remain after removing IDs.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Ensure categorical values are consistently represented as strings.
    for col in categorical_features:
        X[col] = X[col].map(
            lambda value: str(value).strip() if pd.notna(value) else None
        )

    return X, y, numeric_features, categorical_features


# --------------------------------------------------
# 4. Build preprocessing + Random Forest pipeline
# --------------------------------------------------
def build_pipeline(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# --------------------------------------------------
# 5. Train, evaluate, and save model
# --------------------------------------------------
def train_model():
    print("=" * 60)
    print("STUDENT PLACEMENT PREDICTION - MODEL TRAINING")
    print("=" * 60)

    df = load_dataset()
    print(f"\nDataset loaded: {DATA_PATH}")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nPlacement Status counts:")
    print(df[TARGET_COLUMN].value_counts().to_string())

    X, y, numeric_features, categorical_features = prepare_features(df)
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(
        f"Categorical features ({len(categorical_features)}): "
        f"{categorical_features}"
    )

    if len(X) < 5:
        raise ValueError(
            "At least 5 usable student records are recommended "
            "for a train/test evaluation."
        )

    # Stratification keeps class proportions in both sets when possible.
    class_counts = y.value_counts()
    stratify_target = (
        y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    evaluation_model = build_pipeline(
        numeric_features, categorical_features
    )
    print("\nTraining model for evaluation...")
    evaluation_model.fit(X_train, y_train)

    y_pred = evaluation_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (HELD-OUT TEST SET)")
    print("=" * 60)
    print(f"Test-set accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Refit on all rows for the final model used by the Streamlit app.
    print("\nTraining final model on all available records...")
    final_model = build_pipeline(
        numeric_features, categorical_features
    )
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Bundle format: app.py should load bundle["model"] and
    # bundle["feature_columns"] when using this training script.
    model_bundle = {
        "model": final_model,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "classes": final_model.classes_.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "evaluation_accuracy": float(accuracy),
    }
    joblib.dump(model_bundle, MODEL_PATH)

    print("\nTraining completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print("Saved feature columns:", model_bundle["feature_columns"])
    print(
        "\nNote: Accuracy is based on synthetic practice data if the supplied "
        "CSV is the generated sample dataset. It is not a measure of real-world "
        "placement performance."
    )


if __name__ == "__main__":
    train_model()
