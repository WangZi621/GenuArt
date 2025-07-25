import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import base64

# App config
st.set_page_config(
    page_title="Guneart - AI vs Human Art Classifier", 
    page_icon="🎨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
def add_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .title-container {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .title-text {
        color: white;
        font-size: 3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle-text {
        color: #f0f0f0;
        font-size: 1.2rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .upload-container {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        background-color: #f0f8ff;
        border-color: #764ba2;
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
    }
    
    .ai-result {
        border-top: 5px solid #ff6b6b;
    }
    
    .human-result {
        border-top: 5px solid #4ecdc4;
    }
    
    .confidence-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 5px;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
    }
    
    .prediction-text {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Add background pattern
def add_background():
    st.markdown("""
    <style>
    .stApp {
        background-image: radial-gradient(#e0e0e0 1px, transparent 1px);
        background-size: 20px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model once
@st.cache_resource
def load_my_model():
    model_path = "best_model_final_attempt.keras"
    return load_model(model_path)

# Main app
def main():
    add_custom_css()
    add_background()
    
    # Header with gradient background
    st.markdown("""
    <div class="title-container">
        <div class="title-text">🎨 Guneart AI</div>
        <div class="subtitle-text">Discover whether a painting was created by artificial intelligence or a human artist</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About Guneart")
        st.markdown("""
        Guneart uses deep learning to analyze artistic styles and determine if artworks were created by:
        - 🤖 **Artificial Intelligence**
        - 🎨 **Human Artists**
        
        The model recognizes 30 different artistic styles and movements.
        """)
        
        st.markdown("---")
        st.subheader("🎯 How it works")
        st.markdown("""
        1. Upload an image of a painting
        2. Our AI analyzes visual patterns
        3. Get instant classification results
        """)
    
    # Main content
    col1, col2, col3 = st.columns([1,3,1])
    
    with col2:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📤 **Upload an Artwork**", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption="🖼️ Uploaded Artwork", use_column_width=True)
        
        with st.spinner("🔍 Analyzing artwork with Guneart AI..."):
            # Preprocess image
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_expanded = np.expand_dims(img_array, axis=0)
            
            # Predict
            model = load_my_model()
            pred_probs = model.predict(img_expanded)[0]
            predicted_class = class_names[np.argmax(pred_probs)]
            confidence = np.max(pred_probs) * 100
            
            # Determine result type
            is_ai = 'AI' in predicted_class
            
            # Display results
            result_class = "ai-result" if is_ai else "human-result"
            result_emoji = "🤖" if is_ai else "🎨"
            result_text = "**AI Generated Artwork**" if is_ai else "**Human Created Artwork**"
            result_color = "#ff6b6b" if is_ai else "#4ecdc4"
            
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h2>{result_emoji} Classification Result</h2>
                <div class="prediction-text" style="color: {result_color};">{result_text}</div>
                <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%"></div>
                </div>
                <p><small>Predicted Style: <code>{predicted_class}</code></small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by <strong>Guneart AI</strong> | Advanced Deep Learning for Art Classification</p>
    </div>
    """, unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()
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
