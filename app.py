import os
import sys
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Ensure backward compatibility for unpickling tokenizer across Keras versions
sys.modules['keras.preprocessing'] = tf.keras.preprocessing
sys.modules['keras.preprocessing.text'] = tf.keras.preprocessing.text
sys.modules['keras.preprocessing.sequence'] = tf.keras.preprocessing.sequence

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    concatenate,
    Bidirectional,
    Dot,
    Activation,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="CaptionCraft-AI | Image Caption Generator",
    page_icon="📷",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .caption-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.05), rgba(15, 23, 42, 0.1));
        border-left: 6px solid #FF4B4B;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .caption-text {
        font-size: 1.25rem;
        font-weight: 500;
        font-style: italic;
        color: #1E293B;
        margin: 0;
    }
    .metric-badge {
        display: inline-block;
        background-color: #EEF2F6;
        color: #334155;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MobileNetV2 feature extractor...")
def load_feature_extractor():
    base_model = MobileNetV2(weights="imagenet")
    feature_extractor = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )
    return feature_extractor


@st.cache_resource(show_spinner="Loading Caption Generator model...")
def load_caption_model(model_path='mymodel.h5'):
    # Attempt 1: Direct load (works on Keras 2)
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception:
        pass

    # Attempt 2: Reconstruct exact functional architecture and load weights (Keras 2 & 3 compatible)
    inputs1 = Input(shape=(1280,), name='input_3')
    fe1 = Dropout(0.5, name='dropout')(inputs1)
    fe2 = Dense(256, activation='relu', name='dense')(fe1)
    fe2_projected = RepeatVector(34, name='repeat_vector')(fe2)
    fe2_projected = Bidirectional(LSTM(256, return_sequences=True), name='bidirectional')(fe2_projected)

    inputs2 = Input(shape=(34,), name='input_4')
    se1 = Embedding(8768, 256, mask_zero=True, name='embedding')(inputs2)
    se2 = Dropout(0.5, name='dropout_1')(se1)
    se3 = Bidirectional(LSTM(256, return_sequences=True), name='bidirectional_1')(se2)

    attention = Dot(axes=[2, 2], name='dot')([fe2_projected, se3])
    attention_scores = Activation('softmax', name='activation')(attention)
    attention_context = Lambda(
        lambda x: tf.einsum('ijk,ijl->ikl', x[0], x[1]),
        name='lambda'
    )([attention_scores, se3])
    context_vector = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name='tf.math.reduce_sum'
    )(attention_context)

    decoder_input = concatenate([context_vector, fe2], axis=-1, name='concatenate')
    decoder1 = Dense(256, activation='relu', name='dense_1')(decoder_input)
    outputs = Dense(8768, activation='softmax', name='dense_2')(decoder1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.load_weights(model_path)
    return model


@st.cache_resource(show_spinner="Loading vocabulary tokenizer...")
def load_tokenizer(tokenizer_path='tokenizer.pkl'):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


# Initialize models and tokenizer
mobilenet_model = load_feature_extractor()
caption_model = load_caption_model('mymodel.h5')
tokenizer = load_tokenizer('tokenizer.pkl')

# Reverse lookup dictionary for ultra-fast word decoding
index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}


def predict_caption(model, image_features, max_length=34):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_idx = int(np.argmax(yhat))
        predicted_word = index_to_word.get(predicted_idx)

        if predicted_word is None or predicted_word == "endseq":
            break

        in_text += " " + predicted_word

    # Clean special tokens
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption.capitalize()


# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/camera.png", width=120)
    st.title("CaptionCraft-AI")
    st.markdown("### ℹ️ About")
    st.write(
        "**CaptionCraft-AI** is an intelligent multimodal system that pairs pre-trained "
        "**MobileNetV2** visual representations with a **Bidirectional LSTM** sequence "
        "decoder featuring an **attention mechanism**."
    )
    st.markdown("---")
    st.markdown("### 📊 Specifications")
    st.markdown(
        "- **Vision Backbone**: MobileNetV2 (ImageNet)\n"
        "- **Feature Vector**: 1,280 dimensions\n"
        "- **Sequence Model**: Attention Bidirectional LSTM\n"
        "- **Dataset**: Flickr8k (8,091 images)\n"
        "- **Vocabulary Size**: 8,768 unique tokens\n"
        "- **Max Caption Length**: 34 words"
    )
    st.markdown("---")
    st.markdown(
        "Built by **[Vishnu Kamatham](https://github.com/vishnukamatham/CaptionCraft-AI)**"
    )


# Main Interface
st.title("📷 CaptionCraft-AI")
st.markdown(
    "Upload any photograph, and CaptionCraft-AI will analyze its contents and synthesize an informative, natural language caption."
)

uploaded_file = st.file_uploader(
    "Drag & drop or browse an image file",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Uploaded Image")
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True, caption=uploaded_file.name)

    with col2:
        st.subheader("✨ Generated Caption")
        with st.spinner("Extracting features and generating description..."):
            # Resize and preprocess image
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_array)

            # Feature extraction
            image_features = mobilenet_model.predict(img_preprocessed, verbose=0)

            # Caption prediction
            caption = predict_caption(caption_model, image_features, max_length=34)

        # Display result with sleek formatting
        st.markdown(
            f"""
            <div class="caption-box">
                <p class="caption-text">“{caption}.”</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span class="metric-badge">Image Size: {img.size[0]}×{img.size[1]}px</span>'
            f'<span class="metric-badge">Tokens: {len(caption.split())} words</span>',
            unsafe_allow_html=True
        )
else:
    st.info("👆 Upload a JPG or PNG image above to see the caption generator in action!")
