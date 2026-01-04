import streamlit as st
import random
from datetime import datetime
from transformers import pipeline
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# Load Hugging Face Emotion Model
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion_model = load_emotion_model()

def detect_emotion_ai(text):
    result = emotion_model(text)[0]
    label = result["label"].lower()

    label_map = {
        "anger": "anger",
        "disgust": "anger",
        "fear": "fear",
        "joy": "joy",
        "neutral": "neutral",
        "sadness": "sadness",
        "surprise": "surprise"
    }

    love_words = [
        "love", "propose", "married", "engaged",
        "relationship", "crush", "girlfriend", "boyfriend"
    ]

    if any(word in text.lower() for word in love_words):
        return {"label": "love", "score": 0.95}

    return {
        "label": label_map.get(label, "neutral"),
        "score": result["score"]
    }

# --------------------------------------------------
# Spotify API Setup (Secrets Based)
# --------------------------------------------------
SPOTIFY_CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]

sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

SPOTIFY_QUERY_MAP = {
    "joy": "happy upbeat energetic songs",
    "sadness": "sad emotional songs",
    "anger": "calm anger management music",
    "fear": "peaceful meditation background",
    "love": "romantic acoustic love songs",
    "surprise": "uplifting pop songs",
    "neutral": "chill lofi focus music"
}

def get_spotify_full_track(emotion):
    query = SPOTIFY_QUERY_MAP.get(emotion, "relaxing background music")
    results = sp.search(q=query, limit=10, type="track")
    tracks = results.get("tracks", {}).get("items", [])

    if tracks:
        track = random.choice(tracks)
        embed_url = f"https://open.spotify.com/embed/track/{track['id']}"
        return embed_url, track["name"], track["artists"][0]["name"]

    return None, None, None

# --------------------------------------------------
# Emotion Configuration
# --------------------------------------------------
EMOTION_CONFIG = {
    "joy": {
        "color": "#FFE066",
        "emoji": "😊",
        "message": "Keep shining with joy 🌞",
        "tip": "Share your happiness!"
    },
    "sadness": {
        "color": "#A9CCE3",
        "emoji": "😢",
        "message": "Every storm passes 🌈",
        "tip": "Talk to someone you trust."
    },
    "anger": {
        "color": "#F1948A",
        "emoji": "😠",
        "message": "Take a deep breath 🕊️",
        "tip": "Pause before reacting."
    },
    "fear": {
        "color": "#C39BD3",
        "emoji": "😨",
        "message": "You are stronger than your fears 💪",
        "tip": "Face small fears daily."
    },
    "love": {
        "color": "#F5B7B1",
        "emoji": "❤️",
        "message": "Keep spreading love ❤️",
        "tip": "Cherish your relationships."
    },
    "surprise": {
        "color": "#F9E79F",
        "emoji": "😮",
        "message": "Life is full of surprises ✨",
        "tip": "Embrace change."
    },
    "neutral": {
        "color": "#D5DBDB",
        "emoji": "🙂",
        "message": "Stay balanced 🌿",
        "tip": "Maintain good routines."
    }
}

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="Mindful Moments – AI Emotion Therapy",
    page_icon="🧠"
)

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<div style="background:linear-gradient(135deg,#667eea,#764ba2);
padding:1.8rem;border-radius:15px;color:white;text-align:center;">
<h1>🧠 Mindful Moments – AI Emotion Therapy</h1>
<p>🤖 Hugging Face AI • 🎵 Spotify Music • 🎨 Color Therapy</p>
</div>
""", unsafe_allow_html=True)

st.subheader("💭 Share Your Feelings")
user_text = st.text_area(
    "How do you feel today?",
    placeholder="Type your feelings here..."
)

if st.button("🎭 Detect Emotion"):
    if user_text.strip():
        with st.spinner("Analyzing your emotion..."):
            result = detect_emotion_ai(user_text)
            st.session_state.current_emotion = result["label"]
            st.session_state.history.append({
                "time": datetime.now(),
                "emotion": result["label"],
                "score": result["score"]
            })
            st.success("Emotion Detected Successfully!")
    else:
        st.warning("Please enter some text.")

# --------------------------------------------------
# Display Result
# --------------------------------------------------
if "current_emotion" in st.session_state:
    emotion = st.session_state.current_emotion
    cfg = EMOTION_CONFIG[emotion]

    st.markdown(
        f"""
        <div style="background:{cfg['color']};
        padding:2rem;border-radius:15px;">
        <h2>{cfg['emoji']} {emotion.title()}</h2>
        <p><b>{cfg['message']}</b></p>
        <p>💡 {cfg['tip']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    embed_url, song, artist = get_spotify_full_track(emotion)
    if embed_url:
        st.markdown(f"### 🎵 {song} – *{artist}*")
        st.markdown(
            f"""
            <iframe src="{embed_url}" width="100%" height="80"
            frameborder="0"
            allow="autoplay; clipboard-write; encrypted-media;
            fullscreen; picture-in-picture"></iframe>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# Emotion History
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 Recent Emotions")

for entry in st.session_state.history[-5:][::-1]:
    st.write(
        f"{entry['time'].strftime('%H:%M:%S')} – "
        f"{entry['emotion'].title()} ({entry['score']:.2f})"
    )
