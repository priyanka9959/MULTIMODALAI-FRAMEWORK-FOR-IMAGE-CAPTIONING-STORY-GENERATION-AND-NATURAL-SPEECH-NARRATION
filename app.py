import streamlit as st
from PIL import Image
import tempfile
import os
import shutil
import re
import base64
import numpy as np
from image_captioning_story_pipeline import (
    load_captions,
    extract_features,
    generate_caption,
    generate_story_from_caption,
    story_to_audio,
    define_model,
    Tokenizer,
    max_caption_length,
    create_sequences
)
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Fix for pad_sequences error

# Helper to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text).strip()

# Helper to remove repeated words in captions
def clean_caption(text):
    words = text.strip().split()
    cleaned = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i - 1]:
            cleaned.append(word)
    return ' '.join(cleaned)

# Initialize session state
if 'caption' not in st.session_state:
    st.session_state.caption = None
if 'story' not in st.session_state:
    st.session_state.story = None
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

# Load models with caching
@st.cache_resource
def load_models():
    captions_file = 'Flickr8k_text/Flickr8k.token.txt'
    if not os.path.exists(captions_file):
        st.error(f"Caption file not found at: {captions_file}")
        return None

    captions = load_captions(captions_file, limit=100)
    all_captions = [cap for caps in captions.values() for cap in caps]

    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max_caption_length(captions)

    model = define_model(vocab_size, max_length)

    try:
        model.load_weights('model_weights.weights.h5')
    except Exception as e:
        st.warning("⚠️ Failed to load trained model weights. Captions may be poor quality.")

    return {
        "tokenizer": tokenizer,
        "model": model,
        "max_length": max_length,
        "vocab_size": vocab_size
    }

# UI title and instructions
st.title("📷 Image Story Generator")
st.markdown("""
Upload an image to generate a caption, then create a story based on it, 
and finally listen to or download the audio version.
""")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    story_length = st.slider("Story length", 100, 500, 250)
    temperature = st.slider("Creativity", 0.1, 1.0, 0.9)
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("1. Upload an image")
    st.markdown("2. Get an AI-generated caption")
    st.markdown("3. Generate a story based on the caption")
    st.markdown("4. Listen to or download the audio version")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, "temp_image.jpg")

    try:
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = Image.open(temp_image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image.close()

        models = load_models()
        if models is None:
            st.error("Model loading failed.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            st.stop()

        # Generate caption
        if st.button("Generate Caption"):
            with st.spinner("Analyzing image and generating caption..."):
                try:
                    image_id = os.path.splitext(os.path.basename(temp_image_path))[0]
                    features = {image_id: None}
                    features = extract_features(temp_dir, features.keys())

                    if not features:
                        st.error("Failed to extract features.")
                        raise Exception("Feature extraction failed.")

                    photo = list(features.values())[0].reshape((1, 4096))
                    raw_caption = generate_caption(
                        models["model"], 
                        models["tokenizer"], 
                        photo, 
                        models["max_length"]
                    )
                    st.session_state.caption = clean_caption(raw_caption)

                    st.success("Caption generated successfully!")
                    st.markdown(f"**Caption:** {st.session_state.caption}")
                except Exception as e:
                    st.error(f"Caption error: {str(e)}")

        # Generate story
        if st.session_state.caption:
            if st.button("Generate Story"):
                with st.spinner("Creating an imaginative story..."):
                    try:
                        raw_story = generate_story_from_caption(
                            st.session_state.caption,
                            max_length=story_length
                        )
                        st.session_state.story = remove_urls(raw_story)

                        st.success("Story generated successfully!")
                        st.markdown("**Story:**")
                        st.write(st.session_state.story)
                    except Exception as e:
                        st.error(f"Story error: {str(e)}")

        # Generate audio
        if st.session_state.story:
            if st.button("Generate Audio"):
                with st.spinner("Converting story to audio..."):
                    try:
                        audio_path = os.path.join(temp_dir, "generated_story.mp3")
                        story_to_audio(st.session_state.story, output_path=audio_path)
                        st.session_state.audio_path = audio_path

                        st.success("Audio generated successfully!")
                        st.audio(audio_path, format="audio/mp3")

                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        b64 = base64.b64encode(audio_bytes).decode()
                        href = f'<a href="data:file/mp3;base64,{b64}" download="generated_story.mp3">Download Audio</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Audio error: {str(e)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

st.markdown("---")

