import streamlit as st


# =====================================
# CUSTOM CSS
# =====================================

def load_css():
    st.markdown("""
    <style>

    /* Application Background */
    .stApp {
        background-color: #F5F7FB;
    }

    /* Main Content */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1100px;
    }

    /* Header */
    .app-header {
        background-color: #1E3A5F;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
    }

    .app-header h1 {
        color: white !important;
        font-size: 28px;
        margin: 0;
    }

    .app-header p {
        color: #E2E8F0 !important;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Section Headings */
    h1, h2, h3 {
        color: #1E3A5F;
    }

    /* Metric Cards */
    .metric-card {
        background-color: white;
        padding: 15px;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 7px;
        font-weight: 600;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        padding: 15px;
        margin-top: 30px;
        border-top: 1px solid #E2E8F0;
    }

    </style>
    """, unsafe_allow_html=True)


# =====================================
# APPLICATION HEADER
# =====================================

def show_header():
    st.markdown("""
    <div class="app-header">
        <h1>Student Placement Prediction</h1>
        <p>
            Machine Learning Application using Python and Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# APPLICATION FOOTER
# =====================================

def show_footer():
    st.markdown("""
    <div class="app-footer">
        Student Placement Prediction |
        Python | Pandas | Scikit-learn | Streamlit
    </div>
    """, unsafe_allow_html=True)
