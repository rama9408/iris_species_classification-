import streamlit as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Iris Species Classifier", layout="wide")

st.title("🌸 Iris Species Classification Dashboard")
st.markdown("This app compares the performance of **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** on the classic Iris dataset.")

# --- DATA PREPARATION ---
@st.cache_data
def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrame for plotting
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = [iris.target_names[i] for i in iris.target]
    
    return X_train_scaled, X_test_scaled, y_train, y_test, iris.target_names, iris_df

X_train, X_test, y_train, y_test, target_names, iris_df = load_and_preprocess_data()

# --- MODEL TRAINING & EVALUATION ---
# 1. SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# 2. KNN Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# --- SIDEBAR USER INPUT (Bonus Feature) ---
st.sidebar.header("🔮 Real-time Prediction")
st.sidebar.write("Slide to test custom flower measurements:")
sepal_l = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_w = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_l = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_w = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Build custom input payload
custom_sample = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
# We need a dedicated scaler for un-split data to scale custom inputs correctly
full_scaler = StandardScaler().fit(load_iris().data)
custom_sample_scaled = full_scaler.transform(custom_sample)

# --- SIDEBAR PREDICTIONS ---
st.sidebar.subheader("Results")
pred_svm_custom = target_names[svm_model.predict(custom_sample_scaled)[0]]
pred_knn_custom = target_names[knn_model.predict(custom_sample_scaled)[0]]
st.sidebar.write(f"**SVM Predicts:** `{pred_svm_custom}`")
st.sidebar.write(f"**KNN Predicts:** `{pred_knn_custom}`")

# --- MAIN LAYOUT: TWO SEPARATE COLUMNS ---
col1, col2 = st.columns(2)

# --- COLUMN 1: SVM Metrics ---
with col1:
    st.header("⚡ Support Vector Machine (Linear)")
    
    # Metric Callout
    svm_acc = accuracy_score(y_test, y_pred_svm)
    st.metric(label="Test Accuracy", value=f"{svm_acc * 100:.2f}%")
    
    # Classification Report
    st.subheader("Classification Report")
    svm_rep = classification_report(y_test, y_pred_svm, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(svm_rep).transpose().iloc[:-1, :3].style.format("{:.2f}"))
    
    # Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    fig_svm, ax_svm = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax_svm, cbar=False)
    ax_svm.set_xlabel('Predicted labels')
    ax_svm.set_ylabel('True labels')
    st.pyplot(fig_svm)

# --- COLUMN 2: KNN Metrics ---
with col2:
    st.header("🎯 K-Nearest Neighbors (k=3)")
    
    # Metric Callout
    knn_acc = accuracy_score(y_test, y_pred_knn)
    st.metric(label="Test Accuracy", value=f"{knn_acc * 100:.2f}%")
    
    # Classification Report
    st.subheader("Classification Report")
    knn_rep = classification_report(y_test, y_pred_knn, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(knn_rep).transpose().iloc[:-1, :3].style.format("{:.2f}"))
    
    # Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    fig_knn, ax_knn = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, yticklabels=target_names, ax=ax_knn, cbar=False)
    ax_knn.set_xlabel('Predicted labels')
    ax_knn.set_ylabel('True labels')
    st.pyplot(fig_knn)

---
# --- VISUALIZATION SECTION ---
st.markdown("---")
st.header("📊 Data Exploration: Feature Relationships")
st.write("Below is the pair plot showing how different features isolate the various Iris species.")

# Generate and display pairplot safely
with st.spinner("Generating Pair Plot..."):
    pair_plot = sns.pairplot(iris_df, hue='species', palette='Set2')
    st.pyplot(pair_plot.fig)
