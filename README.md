# MULTIMODALAI-FRAMEWORK-FOR-IMAGE-CAPTIONING-STORY-GENERATION-AND-NATURAL-SPEECH-NARRATION

This project uses deep learning models to generate captions and stories from images, followed by text-to-speech conversion. It combines computer vision and natural language processing to bring images to life through storytelling.

---

## 📌 Features

- 🔍 **Image Feature Extraction** using InceptionV3
- 🧠 **Caption Generation** using a trained LSTM model
- 📖 **Story Generation** using GPT-2
- 🔊 **Text-to-Speech (TTS)** using Google Text-to-Speech (gTTS)
- 🖼️💬 **Interactive Interface** built with Streamlit

---

## 🧰 Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **Transformers (Hugging Face GPT-2)**
- **gTTS**
- **Streamlit**
- **Flickr8k Dataset** for training
- **NumPy, Matplotlib, Pickle** and other essentials

---

## 🚀 How It Works

1. **Image Upload:** The user uploads an image via the Streamlit interface.
2. **Image Feature Extraction:** InceptionV3 extracts a 2048-dimensional vector from the image.
3. **Caption Generation:** A trained LSTM model predicts a suitable caption.
4. **Story Generation:** GPT-2 generates a story based on the caption.
5. **Text-to-Speech:** gTTS converts the story into audio.
6. **Final Output:** The image, caption, story, and audio are presented to the user.

---

## 📁 Project Structure

├── app.py # Streamlit interface
├── image_captioning.py # LSTM-based caption generation
├── story_generator.py # GPT-2 story generation
├── tts.py # Text-to-Speech with gTTS
├── feature_extractor.py # InceptionV3 model for image features
├── utils.py # Helper functions
├── model/ # Trained models and tokenizers
├── data/ # Sample images and dataset (Flickr8k)
└── README.md


---

## 🧪 Demo

> 📌 *Upload an image → Get a caption and a story → Listen to the story come alive!*  
*(You can include Streamlit app screenshots or video links here)*

---

## 📚 Future Scope

- ✨ Use CLIP or BLIP-2 for better captioning
- 🗣️ Improve story coherence using larger LLMs like GPT-3 or GPT-4
- 🌐 Deploy as a web app using Hugging Face Spaces or Streamlit Cloud
- 🧠 Support multilingual storytelling and emotion-based narration

---

## 🧑‍💻 Author
Priyanka Gundeboyena  
Final Year B.Tech in Data Science  

---

## 📜 License

This project is licensed under the MIT License.

---

## 🌟 Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Hugging Face Transformers](https://huggingface.co/)
- [Google Text-to-Speech](https://pypi.org/project/gTTS/)
- [Streamlit](https://streamlit.io/)


