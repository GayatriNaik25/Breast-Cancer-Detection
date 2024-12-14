import pickle
import streamlit as st
import numpy as np
import base64
from PIL import Image

# Function to set a background image and sidebar color
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        encoded_image = base64.b64encode(data).decode()

    page_bg_img = page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    /* Change the sidebar background color to black */
    section[data-testid="stSidebar"] {{
        background-color: #000000; /* Black color */
        color: white; /* Sidebar text color in white for contrast */
    }}
    /* Sidebar text styles */
    section[data-testid="stSidebar"] .css-1y4p8pa, 
    section[data-testid="stSidebar"] .css-17eq0hr {{
        color: white !important; /* Ensure sidebar text remains white */
    }}
    /* General text styles */
    h1, h2, h3, h4, h5, h6, p, label {{
        color: white;
    }}
    /* Custom button styles */
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
        color: white;
    }}
    /* Slider label styles */
    .css-1q8dd3e, .css-1b9lf8j {{
        color: white !important;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Set background image and sidebar style
set_background(r"C:\Users\new\Downloads\image (1).png")

# Load scaler and model paths
scaler = pickle.load(open(r"C:\Users\new\Project\Data Science\BREAST CANCER APP\scaler.pkl", "rb"))

model_files = {
    'K-Nearest Neighbors': r"C:\Users\new\Project\Data Science\BREAST CANCER APP\knn_breastcancer.pkl",
    'Logistic Regression': r"C:\Users\new\Project\Data Science\BREAST CANCER APP\logistic_regression_breastcancer.pkl",
    'Decision Tree': r"C:\Users\new\Project\Data Science\BREAST CANCER APP\decision_tree_breastcancer.pkl",
    'Random Forest': r"C:\Users\new\Project\Data Science\BREAST CANCER APP\random_forest_breastcancer.pkl",
    'SVC': r"C:\Users\new\Project\Data Science\BREAST CANCER APP\svc_breastcancer.pkl"}
# Title and updated description
st.markdown('<h1 style="text-align: center; color: white;">Breast Cancer Prediction Tool 🩺</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="color: white; text-align: center; font-size: 16px; line-height: 1.5;">
Welcome to the Breast Cancer Prediction Tool, an AI-powered application designed to assist healthcare professionals. 
Using advanced machine learning models, this tool predicts whether a tumor is likely to be <b>benign</b> or <b>malignant</b>. 
<br><br>
""", unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.header("Choose Classifier")
classifier_name = st.sidebar.selectbox("Select the classifier for prediction", list(model_files.keys()))

# Load the selected model
model = pickle.load(open(model_files[classifier_name], 'rb'))

# Sidebar for classifier information
classifier_descriptions = {
    'K-Nearest Neighbors': (
        "K-Nearest Neighbors (KNN) is a simple, intuitive algorithm that classifies data points based on the "
        "classes of their nearest neighbors. It is especially effective for small datasets with distinct boundaries."
    ),
    'Logistic Regression': (
        "Logistic Regression predicts the probability that a data point belongs to a certain class. "
        "It uses a sigmoid function to model the probability of binary outcomes, making it ideal for classification tasks."
    ),
    'Decision Tree': (
        "Decision Trees are intuitive models that split data into branches based on feature values. "
        "Each branch represents a decision, and the leaves of the tree represent the predicted outcomes."
    ),
    'Random Forest': (
        "Random Forest builds multiple decision trees during training and aggregates their predictions. "
        "This ensemble method improves accuracy and reduces overfitting, making it robust for various datasets."
    ),
    'SVC': (
        "Support Vector Classification (SVC) identifies a hyperplane that best separates classes in a high-dimensional space. "
        "It works well for datasets where the separation between classes is clear and margins can be maximized."
    )
}
st.sidebar.subheader(f"About {classifier_name}")
st.sidebar.write(classifier_descriptions[classifier_name])

# Function to get user input
def get_user_input():
    st.markdown("<h3 style='color:white;'>Input Tumor Characteristics</h3>", unsafe_allow_html=True)
    clump_thickness = st.slider("Clump Thickness", 1, 10)
    uniformity_cell_size = st.slider("Uniformity Cell Size", 1, 10)
    uniformity_cell_shape = st.slider("Uniformity Cell Shape", 1, 10)
    marginal_adhesion = st.slider("Marginal Adhesion", 1, 10)
    single_epithelial_cell_size = st.slider("Single Epithelial Cell Size", 1, 10)
    bare_nuclei = st.slider("Bare Nuclei", 1, 10)
    bland_chromatin = st.slider("Bland Chromatin", 1, 10)
    normal_nucleoli = st.slider("Normal Nucleoli", 1, 10)
    mitoses = st.slider("Mitoses", 1, 10)

    return [clump_thickness, uniformity_cell_size, uniformity_cell_shape, 
            marginal_adhesion, single_epithelial_cell_size, bare_nuclei, 
            bland_chromatin, normal_nucleoli, mitoses]

# Function to make predictions
def make_prediction(model, input_data):
    scaled_input = scaler.transform([input_data])
    return model.predict(scaled_input)

# Display prediction result
def display_result(prediction):
    if prediction[0] == 2:  # Assuming '2' is benign
        st.success("The predicted class is **Benign (Non-cancerous)**.", icon="✅")
    else:  # Assuming '4' is malignant
        st.error("The predicted class is **Malignant (Cancerous)**.", icon="⚠️")

# Get user input and make prediction
input_data = get_user_input()
if st.button("Predict"):
    prediction = make_prediction(model, input_data)
    display_result(prediction)

# Sidebar additional info
st.sidebar.markdown("[Learn more about classifiers](https://scikit-learn.org/stable/supervised_learning.html)")