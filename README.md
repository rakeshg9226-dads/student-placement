# Student Placement Prediction System --- Machine Learning & Streamlit

A beginner-friendly Python project that demonstrates student placement
classification using a Random Forest model and a Streamlit web
application. Users can review student data, view analytics, enter
student details to obtain a model prediction, and search or download
records.

> **Important:** The included practice dataset contains 500 synthetic
> records and 13 columns. It is for learning and demonstration only.
> Predictions and metrics are not evidence of real-world placement
> performance and must not be used as the sole basis for hiring,
> admissions, or other high-impact decisions.

## Features

-   Loads student records from a CSV file.
-   Shows total records, total columns, and missing-value count.
-   Displays a preview of the dataset.
-   Provides a Placement Analytics page using the charts implemented in
    `chart.py`.
-   Loads a saved model from `models/placement_model.pkl`.
-   Builds prediction inputs using the model's feature columns.
-   Predicts the `Placement Status` label.
-   Displays prediction probabilities when supported by the model.
-   Searches across dataset columns.
-   Downloads the filtered records as a CSV file.
-   Uses separate modules for data loading, charts, and styling.

## Technologies Used

-   Python
-   Streamlit
-   Pandas
-   Scikit-learn
-   Random Forest Classifier
-   Joblib
-   CSV
-   CSS styling through Streamlit

## Dataset

Expected dataset path:

``` text
data/student_placement.csv
```

Expected columns:

-   `Student ID`
-   `CGPA`
-   `IQ`
-   `Internships`
-   `Projects`
-   `Communication Skills`
-   `Technical Skills`
-   `Aptitude Score`
-   `Attendance (%)`
-   `Backlogs`
-   `Degree`
-   `Branch`
-   `Placement Status`

The target column is `Placement Status`, with labels such as `Placed`
and `Not Placed`. The supplied dataset is synthetic practice data. If
you replace it with another CSV, confirm that its column names and data
types are compatible with the training script and application.

## Project Structure

``` text
Student_Placement_Prediction/
│
├── app.py
├── train_model.py
├── data_loader.py
├── chart.py
├── style.py
├── requirements.txt
├── README.md
├── Project_Synopsis.md
│
├── data/
│   └── student_placement.csv
│
└── models/
    └── placement_model.pkl
```

The project files should match your local version. `app.py` imports
`data_loader.py`, `chart.py`, and `style.py`. The `train_model.py`
script is required to train or retrain the model. The model file is
generated during training and is needed by the prediction page.

## Requirements

-   Python 3.9 or newer
-   pip
-   A code editor such as Visual Studio Code
-   A web browser

Core packages:

``` bash
pip install streamlit pandas scikit-learn joblib
```

If the project includes `requirements.txt`, install all listed
dependencies from the project folder:

``` bash
pip install -r requirements.txt
```

If `chart.py` imports an additional plotting library, make sure that
library is included in `requirements.txt` and installed.

## Setup Instructions

### 1. Open the project folder

Open the project directory in Visual Studio Code or a terminal.

### 2. Create and activate a virtual environment (recommended)

Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If no `requirements.txt` file is available yet, install the core
packages:

``` bash
pip install streamlit pandas scikit-learn joblib
```

### 4. Check the dataset path

Make sure the CSV file is present at:

``` text
data/student_placement.csv
```

The actual path used by the application is defined in `data_loader.py`.

## Train the Machine Learning Model

If `models/placement_model.pkl` does not exist, train the model before
using the prediction page.

From the project root, run:

``` bash
python train_model.py
```

The training script should prepare the data, train and evaluate the
classifier, and save the trained model bundle. Confirm the output path
in the training script; the application expects:

``` text
models/placement_model.pkl
```

The saved object may contain the fitted model and metadata such as
feature columns and class labels. Keep the training and prediction
preprocessing consistent.

## Run the Streamlit Application

From the project root, run:

``` bash
python -m streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that
address in your browser.

## Application Pages

### 1. Home

Shows project information, total student records, total columns, missing
values, and a dataset preview.

### 2. Placement Analytics

Calls `show_charts(df)` from `chart.py`. The available visualizations
depend on the charts implemented in that file.

### 3. Predict Placement

Loads the trained model, creates input controls from its expected
feature columns, and returns a predicted placement label. If supported
by the model, the page also displays class probabilities.

### 4. Dataset Explorer

Allows you to search across dataset values, view matching records, and
download the filtered data as `student_placement_data.csv`.

## Troubleshooting

### `ModuleNotFoundError`

Install the missing package in the same Python environment used to run
Streamlit. For example:

``` bash
python -m pip install streamlit pandas scikit-learn joblib
```

### `streamlit is not recognized`

Use:

``` bash
python -m streamlit run app.py
```

This avoids relying on the standalone `streamlit` command being on the
Windows PATH.

### Dataset cannot be loaded

-   Confirm that `data/student_placement.csv` exists.
-   Check the path configured in `data_loader.py`.
-   Confirm that the CSV is readable and has the expected columns.

### Trained model not found

Run `python train_model.py` and confirm that it saves the model to
`models/placement_model.pkl`.

### Prediction fails or feature columns do not match

-   Retrain the model using the current dataset and training script.
-   Ensure that the model bundle includes the correct feature columns.
-   Keep preprocessing consistent between training and prediction.
-   Check that all required project files and packages are available.

### Analytics charts do not appear

Review `chart.py` and install any plotting packages it imports. The main
app displays an error message if `show_charts(df)` fails.

## Uploading to GitHub

Before uploading:

-   Include source files, `README.md`, `Project_Synopsis.md`, and
    `requirements.txt`.
-   Include the dataset only if you have permission to share it and it
    contains no private information.
-   Do not upload passwords, API keys, private credentials, or
    confidential data.
-   The trained `.pkl` model may be included if appropriate for your
    repository and file-size limits; otherwise, document how to train
    it.
-   Consider excluding `.venv/`, `__pycache__/`, and other generated or
    environment-specific files using `.gitignore`.

Example `.gitignore`:

``` gitignore
.venv/
__pycache__/
*.py[cod]
.env
.streamlit/secrets.toml
```

If the repository is public, verify that the dataset and all included
files are safe to share.

## Responsible Use

This is an educational demonstration based on synthetic data. Do not
treat its predictions as definitive assessments of an individual
student. Any real-world use would require representative data,
appropriate consent and privacy controls, rigorous validation, fairness
assessment, and human oversight.

## Author / Project Information

**Project:** Student Placement Prediction System\
**Category:** Machine Learning and Data Analysis\
**Interface:** Streamlit\
**Model:** Random Forest Classifier
