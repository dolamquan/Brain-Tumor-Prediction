import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('tumor_classification_model.keras')

# Define image size and class labels
IMG_SIZE = (128, 128)
class_names = ['glioma', 'meningioma', 'pituitary', 'no tumor']

# Streamlit app title and description
st.title("Brain MRI Tumor Classification")
st.write("Upload an MRI image to classify whether it contains a tumor and, if so, what type of tumor.")

# File uploader for MRI images
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the image and make predictions
def classify_image(image_path):
    # Load the image and resize it to the required size
    img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize(IMG_SIZE)
    
    # Convert image to numpy array and preprocess it for the model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    
    # Make prediction using the loaded model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get index of the highest probability class
    confidence = np.max(predictions[0]) * 100    # Get the confidence of the prediction
    
    return class_names[predicted_class], confidence

# Check if an image has been uploaded
if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Perform classification when the user clicks the "Classify" button
        if st.button('Classify'):
            tumor_type, confidence = classify_image("temp_image.jpg")
            
            # Display the classification result
            st.write(f"**Prediction:** {tumor_type}")
            st.write(f"**Confidence:** {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    st.write("Please upload an MRI image to proceed.")

