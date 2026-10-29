# Student Placement Prediction System Using Machine Learning

## Project Synopsis

### Project Title

**Student Placement Prediction System Using Machine Learning and
Streamlit**

------------------------------------------------------------------------

# 1. Introduction

Student placement is an important stage in a student's academic journey.
Academic performance, internships, projects, communication skills,
technical skills, aptitude, attendance, and backlogs may be useful
indicators when studying placement outcomes.

The Student Placement Prediction System is a Python-based
machine-learning project that demonstrates how student data can be
explored and used to estimate a placement outcome. It combines a Random
Forest classification model with a Streamlit web application. Users can
view dataset summaries, explore charts, enter student information,
receive a model prediction, and download filtered records.

The supplied practice dataset contains 500 records and 13 columns. It is
synthetic data for learning and demonstration, not evidence of
real-world placement outcomes.

------------------------------------------------------------------------

# 2. Objectives of the Project

The main objectives are:

-   Load and validate student data from a CSV file.
-   Clean dataset column names and remove empty or duplicate records.
-   Prepare numeric and categorical features for machine learning.
-   Train a Random Forest Classifier to predict the `Placement Status`
    label.
-   Evaluate the model using suitable classification metrics.
-   Save the trained model for reuse by the application.
-   Display student dataset summaries and analytics charts.
-   Provide a form for entering student details and generating
    predictions.
-   Allow users to search, inspect, and download dataset records.

------------------------------------------------------------------------

# 3. Problem Statement

Reviewing student records manually can make it difficult to explore
patterns across academic, skill-related, and background attributes. A
simple, structured application can help users inspect a dataset and
demonstrate how a machine-learning model produces a placement prediction
from selected features.

This project addresses the problem by bringing data loading,
preprocessing, model training, evaluation, visualization, and prediction
into one beginner-friendly application. The prediction is an estimate
from the trained model and must not be treated as a guarantee or as the
sole basis for hiring, admissions, or other high-impact decisions.

------------------------------------------------------------------------

# 4. Scope of the Project

The project is intended for educational use, machine-learning practice,
and demonstration of a data-driven application.

Potential uses include:

-   Exploring sample student placement data.
-   Understanding relationships between student attributes and placement
    labels.
-   Practising data preprocessing and classification.
-   Demonstrating predictions through a web interface.
-   Learning how to connect a saved model to a Streamlit application.

The project can be extended with representative data collected with
appropriate permission, additional model comparisons, explainability,
improved validation, and monitoring. Real-world use would require
careful assessment of data quality, fairness, privacy, and model
performance.

------------------------------------------------------------------------

# 5. Technology Used

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  Pandas                              Data loading and manipulation

  Scikit-learn                        Data preprocessing, model training,
                                      and evaluation

  Random Forest Classifier            Classification algorithm

  Joblib                              Saving and loading the trained
                                      model

  Streamlit                           Interactive web application

  Matplotlib / Plotly (as implemented Data visualization
  in `chart.py`)                      

  CSV                                 Dataset storage

  HTML / CSS                          Application styling through
                                      Streamlit and `style.py`
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 6. Software Requirements

## Operating System

-   Windows 10 / Windows 11
-   Linux
-   macOS

## Software

-   Python 3.9 or newer
-   Visual Studio Code, PyCharm, or another Python editor
-   Command Prompt or terminal
-   A web browser

## Python Packages

The core packages include:

``` bash
pip install streamlit pandas scikit-learn joblib
```

Install any additional charting packages imported by `chart.py`. If the
project includes a `requirements.txt` file, install the listed
dependencies with:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 7. Dataset Description

Expected dataset path:

`data/student_placement.csv`

The sample dataset contains 500 records and 13 columns.

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  Student ID                          Student identifier; should not be
                                      used as a predictive feature

  CGPA                                Academic performance measure

  IQ                                  IQ score field in the sample data

  Internships                         Number of internships

  Projects                            Number of projects

  Communication Skills                Communication-skill category

  Technical Skills                    Technical-skill category

  Aptitude Score                      Aptitude score

  Attendance (%)                      Attendance percentage

  Backlogs                            Number of backlogs

  Degree                              Degree category

  Branch                              Academic branch

  Placement Status                    Target label, such as `Placed` or
                                      `Not Placed`
  -----------------------------------------------------------------------

**Dataset note:** This dataset is synthetic practice data. Its class
distribution and evaluation results may not represent actual students or
real-world placement outcomes.

------------------------------------------------------------------------

# 8. Project Structure

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

The exact structure can differ slightly by project version. The
application code imports `data_loader.py`, `chart.py`, and `style.py`.
The training script and dataset must be present for retraining. The
model file is generated by the training process and is required for the
prediction page.

------------------------------------------------------------------------

# 9. Module Description

## 9.1 `app.py`

The main Streamlit application. It configures the page, loads the
dataset, displays navigation, and connects the four pages:

-   Home
-   Placement Analytics
-   Predict Placement
-   Dataset Explorer

It also handles errors when the dataset or trained model cannot be
loaded.

## 9.2 `data_loader.py`

Loads the student dataset from the configured CSV path and provides it
to the application.

## 9.3 `chart.py`

Contains the `show_charts(df)` function used to display placement
analytics. The specific charts depend on the implementation in this
file.

## 9.4 `style.py`

Contains styling and layout helpers, including `load_css()`,
`show_header()`, and `show_footer()`.

## 9.5 `train_model.py`

Prepares the dataset, separates input features from the target label,
processes numeric and categorical columns, trains and evaluates a Random
Forest Classifier, and saves the trained model bundle.

## 9.6 `data/student_placement.csv`

Contains the student practice records used for training, analytics, and
exploration.

## 9.7 `models/placement_model.pkl`

Stores the trained model and, depending on the training script, metadata
such as feature names and class labels. This file is loaded by the
prediction page.

------------------------------------------------------------------------

# 10. Methodology

The project follows these main steps:

1.  Load the CSV dataset.
2.  Validate that the target column `Placement Status` exists.
3.  Clean column names and remove empty or duplicate records as
    implemented in the training script.
4.  Separate the target label from the input features.
5.  Exclude student-identification fields from model inputs.
6.  Identify numeric and categorical columns.
7.  Handle missing values using suitable preprocessing.
8.  Encode categorical features for machine learning.
9.  Split the data into training and testing sets.
10. Train a Random Forest Classifier.
11. Evaluate the model with accuracy, a classification report, and a
    confusion matrix.
12. Save the fitted model for reuse.
13. Load the dataset and model in Streamlit.
14. Collect user inputs, generate a prediction, and display the result.
15. Allow users to search and download dataset records.

The exact preprocessing and evaluation steps should match the actual
`train_model.py` file in the project.

------------------------------------------------------------------------

# 11. System Workflow

``` text
Student CSV Dataset
        |
        v
Data Loading and Validation
        |
        v
Data Cleaning and Preprocessing
        |
        v
Training and Testing Split
        |
        v
Random Forest Classifier
        |
        v
Model Evaluation
        |
        v
Save Trained Model
        |
        v
Streamlit Application
        |
        +---- Home and Dataset Summary
        |
        +---- Placement Analytics
        |
        +---- Enter Student Details
        |          |
        |          v
        |     Predict Placement
        |
        +---- Search and Download Dataset
```

------------------------------------------------------------------------

# 12. Expected Output

The application is expected to provide:

-   A home page with student record count, column count, and
    missing-value count.
-   A preview of the dataset.
-   Placement analytics charts implemented in `chart.py`.
-   A student-information form generated from the trained model's
    feature columns.
-   A predicted placement label.
-   Prediction probabilities when the model supports `predict_proba()`.
-   A searchable dataset table.
-   A CSV download for the displayed, filtered records.
-   Clear error messages when required data or model files are missing.

Actual prediction results depend on the dataset, preprocessing, trained
model, and entered values.

------------------------------------------------------------------------

# 13. Testing

The following checks can be performed:

-   Verify that the CSV file exists at the configured path.
-   Confirm that `Placement Status` is present in the training dataset.
-   Check that the training script completes and saves the model.
-   Confirm that all required Python packages are installed.
-   Open each Streamlit navigation page.
-   Test the prediction form with valid feature values.
-   Confirm that prediction errors are displayed clearly.
-   Search for a value in the Dataset Explorer.
-   Download the filtered dataset and verify the CSV file.
-   Check the app when the dataset or model file is unavailable.

------------------------------------------------------------------------

# 14. Advantages

-   Demonstrates a complete beginner-level machine-learning workflow.
-   Combines data exploration, analytics, and prediction in one
    interface.
-   Uses reusable Python modules.
-   Supports numeric and categorical model features when configured by
    the training pipeline.
-   Provides a simple web interface without requiring users to work
    directly in Python code.
-   Helps practise model evaluation and deployment concepts.

------------------------------------------------------------------------

# 15. Limitations

-   The supplied dataset is synthetic and may not reflect real placement
    conditions.
-   Predictions depend on the quality and representativeness of training
    data.
-   Model probabilities are estimates and are not guarantees.
-   Performance on practice data does not establish real-world
    reliability.
-   The application is not a substitute for human judgement or a
    validated institutional placement process.
-   Model quality can change if the dataset, feature columns, or
    preprocessing pipeline changes.

------------------------------------------------------------------------

# 16. Future Enhancements

-   Add comparisons with Logistic Regression, Decision Tree, and other
    classifiers.
-   Add visualizations for feature distributions and placement-label
    counts.
-   Include model explainability and feature-importance views.
-   Add stronger input validation and clearer form labels.
-   Add automated tests and data-quality checks.
-   Use representative, appropriately consented data for any real-world
    evaluation.
-   Assess model fairness, privacy, and performance before considering
    operational use.
-   Deploy the application to a suitable hosting platform.

------------------------------------------------------------------------

# 17. Conclusion

The Student Placement Prediction System demonstrates how Python, Pandas,
Scikit-learn, Random Forest, Joblib, and Streamlit can be combined to
create a machine-learning application. It supports dataset inspection,
visual analytics, placement prediction, and CSV downloads. The project
is suitable for learning and portfolio demonstration, while its
synthetic dataset and predictions should be treated as educational
examples rather than real-world placement evidence.
