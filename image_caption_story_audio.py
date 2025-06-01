import sys
print(sys.executable)

import os
import string
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from gtts import gTTS
from playsound import playsound

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

from transformers import pipeline


# -----------------------------
# Load and process captions
# -----------------------------
def load_captions(filepath, limit=100):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    descriptions = {}
    for line in lines:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            continue
        image_id, caption = tokens[0].split('.')[0], tokens[1]
        if len(descriptions) >= limit and image_id not in descriptions:
            continue
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        if len(caption.split()) < 3:
            continue
        if image_id not in descriptions:
            descriptions[image_id] = []
        descriptions[image_id].append('startseq ' + caption + ' endseq')
    return descriptions

captions = load_captions('Flickr8k_text/Flickr8k.token.txt', limit=100)


# -----------------------------
# Extract features using VGG16
# -----------------------------
def extract_features(directory, valid_ids):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for img_name in tqdm(os.listdir(directory), desc="Extracting features"):
        image_id = img_name.split('.')[0]
        if image_id not in valid_ids:
            continue
        filename = os.path.join(directory, img_name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[image_id] = feature.flatten()
    return features

features = extract_features('Flickr8k_Dataset/Flicker8k_Dataset/', captions.keys())


# -----------------------------
# Tokenization
# -----------------------------
all_captions = [cap for caps in captions.values() for cap in caps]
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

def max_caption_length(descriptions):
    return max(len(d.split()) for desc in descriptions.values() for d in desc)

max_length = max_caption_length(captions)


# -----------------------------
# Prepare sequences
# -----------------------------
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        photo_feature = photos[key]
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photo_feature)
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# -----------------------------
# Define captioning model
# -----------------------------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = define_model(vocab_size, max_length)


# -----------------------------
# Generate caption
# -----------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace("startseq", "").replace("endseq", "").strip()


# -----------------------------
# Generate story using GPT-2
# -----------------------------
story_generator = pipeline("text-generation", model="gpt2", framework="pt")

def generate_story_from_caption(caption_text, max_length=120):
    prompt = f"Write a short and vivid story based on this caption: \"{caption_text}\". The story begins:\n"
    result = story_generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        temperature=0.9,
        truncation=True,
        pad_token_id=50256
    )
    story_text = result[0]['generated_text']
    return story_text.replace(prompt, "").strip()


# -----------------------------
# Convert story to speech
# -----------------------------
def speak_text(text, filename="story_audio.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    print(f"✅ Audio saved: {filename}")
    playsound(filename)


# -----------------------------
# Run captioning + story + TTS
# -----------------------------
image_id = list(features.keys())[0]
photo = features[image_id].reshape((1, 4096))

caption = generate_caption(model, tokenizer, photo, max_length)
print("\n📝 Caption:", caption)

story = generate_story_from_caption(caption)
print("\n📖 Story:\n", story)

speak_text(story)