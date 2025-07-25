import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# App config
st.set_page_config(page_title="Guneart - AI vs Human Art Classifier", page_icon="🖼️", layout="centered")

# Title and description
st.title("🎨 Guneart: AI vs Human Art Classifier")
st.markdown("""
> *Discover whether a painting was created by artificial intelligence or a human artist.*
""")

st.markdown("---")

# Load the model once
@st.cache_resource
def load_my_model():
    model_path = "best_model_final_attempt.keras"  # Path relative to app directory
    return load_model(model_path)

model = load_my_model()

# Define class names
class_names = [
    'AI_LD_art_nouveau', 'AI_LD_baroque', 'AI_LD_expressionism', 'AI_LD_impressionism', 
    'AI_LD_post_impressionism', 'AI_LD_realism', 'AI_LD_renaissance', 'AI_LD_romanticism',
    'AI_LD_surrealism', 'AI_LD_ukiyo-e', 'AI_SD_art_nouveau', 'AI_SD_baroque', 
    'AI_SD_expressionism', 'AI_SD_impressionism', 'AI_SD_post_impressionism', 
    'AI_SD_realism', 'AI_SD_renaissance', 'AI_SD_romanticism', 'AI_SD_surrealism', 
    'AI_SD_ukiyo-e', 'art_nouveau', 'baroque', 'expressionism', 'impressionism', 
    'post_impressionism', 'realism', 'renaissance', 'romanticism', 'surrealism', 'ukiyo_e'
]

# File uploader
st.subheader("📤 Upload an Artwork")
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing artwork..."):
        # Preprocess image
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        # Predict
        pred_probs = model.predict(img_expanded)[0]
        predicted_class = class_names[np.argmax(pred_probs)]
        confidence = np.max(pred_probs) * 100

        # Classify as AI or Human
        if 'AI' in predicted_class:
            st.success(f"🤖 **This artwork was made by AI.**\n\n_(Confidence: {confidence:.2f}%)_")
        else:
            st.info(f"🎨 **This artwork was made by a Human.**\n\n_(Confidence: {confidence:.2f}%)_")

        # Show exact class
        st.write(f"**Predicted Style Class:** `{predicted_class}`")

st.markdown("---")
st.caption("Powered by **Guneart AI** | Deep Learning Model for Art Classification")

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load the model once
# @st.cache_resource
# def load_my_model():
#     model_path = "best_model_final_attempt.keras"  # Path relative to app directory
#     return load_model(model_path)

# model = load_my_model()

# # Define class names
# class_names = [
#     'AI_LD_art_nouveau', 'AI_LD_baroque', 'AI_LD_expressionism', 'AI_LD_impressionism', 
#     'AI_LD_post_impressionism', 'AI_LD_realism', 'AI_LD_renaissance', 'AI_LD_romanticism',
#     'AI_LD_surrealism', 'AI_LD_ukiyo-e', 'AI_SD_art_nouveau', 'AI_SD_baroque', 
#     'AI_SD_expressionism', 'AI_SD_impressionism', 'AI_SD_post_impressionism', 
#     'AI_SD_realism', 'AI_SD_renaissance', 'AI_SD_romanticism', 'AI_SD_surrealism', 
#     'AI_SD_ukiyo-e', 'art_nouveau', 'baroque', 'expressionism', 'impressionism', 
#     'post_impressionism', 'realism', 'renaissance', 'romanticism', 'surrealism', 'ukiyo_e'
# ]

# # Streamlit UI
# st.title("🎨 AI vs Human Art Classifier")
# st.write("Upload an image of a painting to classify whether it was **made by AI or a human artist**.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#     # Preprocess image
#     img_resized = image.resize((224, 224))
#     img_array = np.array(img_resized) / 255.0
#     img_expanded = np.expand_dims(img_array, axis=0)

#     # Predict
#     pred_probs = model.predict(img_expanded)[0]
#     predicted_class = class_names[np.argmax(pred_probs)]

#     # Classify as AI or Human
#     if 'AI' in predicted_class:
#         st.success("🤖 This artwork was **made by AI**.")
#     else:
#         st.info("🎨 This artwork was **made by a Human**.")

#     # Show exact class
#     st.write(f"**Predicted style class:** `{predicted_class}`")
