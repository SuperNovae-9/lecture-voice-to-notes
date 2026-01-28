import streamlit as st
import tempfile
import re
from faster_whisper import WhisperModel

st.set_page_config(page_title="Lecture Voice-to-Notes", layout="wide")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="int8")

stt_model = load_model()

# ---------- Simple Generators ----------
def make_notes(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    notes = []
    for s in sentences[:25]:
        if len(s.strip()) > 15:
            notes.append("• " + s.strip())
    return "\n".join(notes)

def make_quiz(text):
    keywords = list(set(re.findall(r"\b[A-Za-z]{5,}\b", text)))[:5]
    quiz = []
    for i, k in enumerate(keywords, 1):
        quiz.append(
            f"Q{i}: What is related to '{k}'?\n"
            f"A) Concept\nB) Object\nC) Process\nD) None\n"
            f"Correct: A\n"
        )
    return "\n".join(quiz)

def make_flashcards(text):
    words = list(set(re.findall(r"\b[A-Za-z]{6,}\b", text)))[:6]
    cards = []
    for w in words:
        cards.append(f"Q: What is {w}?\nA: {w} is an important concept discussed in the lecture.\n")
    return "\n".join(cards)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Notes", "Quiz", "Flashcards"])

# ---------- Session State ----------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = ""
if "flash" not in st.session_state:
    st.session_state.flash = ""

# ---------- Home ----------
if page == "Home":
    st.title("Lecture Voice-to-Notes Generator")
    st.write("Upload a lecture audio and convert it into notes, quizzes, and flashcards.")

    audio_file = st.file_uploader("Upload lecture audio (.mp3 or .wav)", type=["mp3", "wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            temp_audio_path = tmp.name

        st.info("Transcribing audio...")
        try:
            segments, info = stt_model.transcribe(temp_audio_path)
            text = " ".join([seg.text for seg in segments]).strip()
            if not text:
                st.error("Audio could not be transcribed. Please try a clearer lecture.")
            else:
                st.session_state.transcript = text
                st.success("Lecture transcribed successfully!")
                with st.expander("View Transcript"):
                    st.text_area("", text, height=220)
        except Exception:
            st.error("Could not process this audio file.")

# ---------- Notes ----------
if page == "Notes":
    st.title("Study Notes")
    if not st.session_state.transcript:
        st.warning("Upload audio on Home page first.")
    else:
        if st.button("Generate Notes"):
            st.session_state.notes = make_notes(st.session_state.transcript)
        if st.session_state.notes:
            st.text_area("", st.session_state.notes, height=350)

# ---------- Quiz ----------
if page == "Quiz":
    st.title("Quiz")
    if not st.session_state.transcript:
        st.warning("Upload audio on Home page first.")
    else:
        if st.button("Generate Quiz"):
            st.session_state.quiz = make_quiz(st.session_state.transcript)
        if st.session_state.quiz:
            st.text_area("", st.session_state.quiz, height=350)

# ---------- Flashcards ----------
if page == "Flashcards":
    st.title("Flashcards")
    if not st.session_state.transcript:
        st.warning("Upload audio on Home page first.")
    else:
        if st.button("Generate Flashcards"):
            st.session_state.flash = make_flashcards(st.session_state.transcript)
        if st.session_state.flash:
            st.text_area("", st.session_state.flash, height=350)
