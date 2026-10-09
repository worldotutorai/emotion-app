import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# --------------------------------------------------
# Emotion → Emoji mapping
# --------------------------------------------------

emotion_to_emoji = {
    "joy": "😊",
    "sadness": "😥",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}


# --------------------------------------------------
# Constants
# --------------------------------------------------

MAX_LENGTH = 40


# --------------------------------------------------
# Load model and preprocessing objects
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("my_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_emotion(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    predicted_class = np.argmax(prediction[0])

    emotion = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = prediction[0][predicted_class]

    emoji = emotion_to_emoji.get(
        emotion,
        "🙂"
    )

    return emotion, emoji, confidence, prediction[0]


# --------------------------------------------------
# App UI
# --------------------------------------------------

st.title("😊 Emotion Detection using LSTM")

st.write(
    "Enter a sentence and the trained LSTM model will "
    "predict the emotion expressed in the text."
)

st.divider()


# --------------------------------------------------
# Text input
# --------------------------------------------------

text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: I am having a wonderful day!",
    height=120
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Predict Emotion",
    use_container_width=True
):

    if text.strip() == "":
        st.warning("Please enter some text first.")

    else:

        emotion, emoji, confidence, probabilities = (
            predict_emotion(text)
        )

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Emotion",
                emotion.capitalize()
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:80px;
                padding:20px;
            ">
                {emoji}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            f"The detected emotion is **{emotion.capitalize()}**"
        )

        st.divider()

        # ------------------------------------------
        # Probability chart
        # ------------------------------------------

        st.subheader("Emotion Probabilities")

        emotions = label_encoder.classes_

        probability_dict = {
            emotion: float(probability)
            for emotion, probability
            in zip(emotions, probabilities)
        }

        st.bar_chart(probability_dict)

        # ------------------------------------------
        # Detailed probabilities
        # ------------------------------------------

        st.subheader("Detailed Results")

        for emotion_name, probability in probability_dict.items():

            emoji_value = emotion_to_emoji.get(
                emotion_name,
                "🙂"
            )

            st.write(
                f"{emoji_value} "
                f"**{emotion_name.capitalize()}**"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                f"{probability:.2%}"
            )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About the Model")

    st.write(
        "This application uses an LSTM-based neural "
        "network for text emotion classification."
    )

    st.write("**Supported emotions:**")

    for emotion, emoji in emotion_to_emoji.items():

        st.write(
            f"{emoji} {emotion.capitalize()}"
        )

    st.divider()

    st.caption(
        "Emotion Detection using RNN-LSTM"
    )