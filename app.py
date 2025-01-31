import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import os

# Suppress TensorFlow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide INFO and WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# Set page config
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, rgb(70 121 228), rgb(191 224 253));
        height: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("LiteCNN4Rice.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class labels
CLASS_NAMES = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Preprocess image
def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

# Prediction function
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return CLASS_NAMES[np.argmax(predictions[0])], np.max(predictions[0])

# Load test data
@st.cache_data
def load_test_data():
    test_dir = "test"  # Update this path if needed

    if not os.path.exists(test_dir):
        st.warning("Test directory not found. Evaluation unavailable.")
        return None, None

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    x_test, y_test = np.vstack([generator[i][0] for i in range(len(generator))]), \
                      np.vstack([generator[i][1] for i in range(len(generator))])
    
    return x_test, y_test

def main():
    st.title("Rice Leaf Disease Classification 🌱")
    st.markdown("""
    ### Upload an image of a rice leaf to diagnose potential diseases
    This AI model detects **Bacterial Blight, Blast, Brown Spot, and Tungro**.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            
            with st.spinner('Analyzing...'):
                prediction, confidence = predict(image)
                
                st.success(f"**Prediction:** {prediction}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                disease_info = {
                    'Bacterialblight': "🦠 Bacterial infection. Use copper-based bactericides.",
                    'Blast': "💨 Fungal disease. Use resistant varieties & silicon fertilizers.",
                    'Brownspot': "🍂 Fungal infection. Improve soil fertility & use fungicides.",
                    'Tungro': "🦠 Viral disease. Control green leafhoppers & use resistant cultivars."
                }
                
                st.markdown(f"**Recommended Action:** {disease_info[prediction]}")

    with col2:
        with st.expander("Model Performance Metrics"):
            st.subheader("Model Evaluation")
            
            # Load test data
            x_test, y_test = load_test_data()
            
            if x_test is not None and y_test is not None and st.button("Generate Evaluation Report"):
                with st.spinner('Processing test data...'):
                    y_pred = model.predict(x_test)
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    y_true_labels = np.argmax(y_test, axis=1)

                    # Confusion Matrix
                    fig, ax = plt.subplots(figsize=(8,6))
                    cm = confusion_matrix(y_true_labels, y_pred_labels)
                    ax.imshow(cm, cmap=plt.cm.Blues)
                    ax.set_xticks(range(len(CLASS_NAMES)))
                    ax.set_yticks(range(len(CLASS_NAMES)))
                    ax.set_xticklabels(CLASS_NAMES, rotation=45)
                    ax.set_yticklabels(CLASS_NAMES)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    
                    for i in range(len(CLASS_NAMES)):
                        for j in range(len(CLASS_NAMES)):
                            ax.text(j, i, cm[i, j], ha='center', va='center', 
                                    color='white' if cm[i, j] > cm.max()/2 else 'black')
                    
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    report = classification_report(y_true_labels, y_pred_labels, target_names=CLASS_NAMES)
                    st.code(report)

        # Sample Predictions
        st.subheader("Example Predictions")
        st.markdown("Sample predictions from the test dataset:")

        TEST_DIR = "test"
        if os.path.exists(TEST_DIR):
            sample_images = []
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(TEST_DIR, class_name)
                if os.path.exists(class_dir):
                    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        selected = np.random.choice(images, min(2, len(images)), replace=False)
                        sample_images.extend([(class_name, os.path.join(class_dir, fname)) for fname in selected])

            cols = st.columns(4)
            for idx, (true_class, img_path) in enumerate(sample_images[:8]):
                with cols[idx % 4]:
                    try:
                        with st.container():
                            image = Image.open(img_path)
                            image.thumbnail((250, 250))
                            
                            squared_image = Image.new("RGB", (200, 200), (255, 255, 255))
                            offset = ((200 - image.width) // 2, (200 - image.height) // 2)
                            squared_image.paste(image, offset)
                            
                            st.image(squared_image, width=150)
                            
                            prediction, _ = predict(Image.open(img_path))
                            color = "green" if prediction == true_class else "red"
                            st.markdown(f"<div style='text-align:center'>"
                                        f"<div style='font-weight:bold'>True: {true_class}</div>"
                                        f"<div style='color:{color}; font-weight:bold'>Pred: {prediction}</div>"
                                        f"</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
        else:
            st.warning("Test directory not found. Sample predictions unavailable.")

if __name__ == "__main__":
    main()
