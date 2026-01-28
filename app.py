import streamlit as st
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel

st.set_page_config(page_title="Lecture Voice-to-Notes", layout="wide")

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    stt = WhisperModel("base", compute_type="int8")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return stt, tokenizer, model

stt_model, tokenizer, gen_model = load_models()

# ---------- Helpers ----------
def generate_with_prompt(prompt, text):
    chunks = [text[i:i+900] for i in range(0, len(text), 900)]
    outputs = []

    for ch in chunks:
        input_text = prompt + "\n\n" + ch
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        out = gen_model.generate(**inputs, max_new_tokens=250, do_sample=False)
        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return "\n".join(outputs)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Notes", "Quiz", "Flashcards"])

# ---------- Session State ----------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "quiz_text" not in st.session_state:
    st.session_state.quiz_text = ""
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""
if "flash_text" not in st.session_state:
    st.session_state.flash_text = ""

# ---------- Home ----------
if page == "Home":
    st.title("Lecture Voice-to-Notes Generator")
    st.write("Upload a lecture audio and turn it into notes, quizzes, and flashcards.")

    audio_file = st.file_uploader("Upload lecture audio (.mp3 or .wav)", type=["mp3", "wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            temp_audio_path = tmp.name

        st.info("Transcribing audio...")

        try:
            segments, info = stt_model.transcribe(temp_audio_path)
            text = " ".join([seg.text for seg in segments])
            st.session_state.transcript = text.strip()

            if not st.session_state.transcript:
                st.error("The audio seems to be empty or unclear. Please upload a clearer lecture.")
            else:
                st.success("Lecture transcribed! Go to Notes / Quiz / Flashcards from the sidebar.")
                with st.expander("View Transcript"):
                    st.text_area("", st.session_state.transcript, height=220)

        except Exception:
            st.error("Could not process this audio file. Please try another lecture recording.")

# ---------- Notes ----------
if page == "Notes":
    st.title("Notes Generator")
    if not st.session_state.transcript:
        st.warning("Upload audio on Home page first.")
    else:
        st.text_area("Transcript", st.session_state.transcript, height=180)
        if st.button("Generate Notes"):
            prompt = (
                "Convert the following lecture into simple, student-friendly notes.\n"
                "Use bullet points and small headings.\n"
                "Avoid robotic or formal language.\n"
                "Write like a helpful classmate explaining."
            )
            st.session_state.notes_text = generate_with_prompt(prompt, st.session_state.transcript)

        if st.session_state.notes_text:
            st.subheader("Study Notes")
            st.text_area("", st.session_state.notes_text, height=260)

# ---------- Quiz ----------
if page == "Quiz":
    st.title("Quiz Generator")
    if not st.session_state.transcript:
        st.warning("Upload audio on Home page first.")
    else:
        if st.button("Generate Quiz"):
            prompt = (
                "Create 5 multiple-choice questions from this lecture.\n"
                "Include options and the correct answer for each.\n"
                "Keep it simple and student-friendly."
            )
            st.session_state.quiz_text = generate_with_prompt(prompt, st.session_state.transcript)

        if st.session_state.quiz_text:
            st.subheader("Generated Quiz")
            st.text_area("", st.session_state.quiz_text, height=350)

# ---------- Flashcards ----------
if page == "Flashcards":
    st.title("Flashcards Generator")
    if not st.session_state.transcript:
        st.warning("Upload audio on Home page first.")
    else:
        if st.button("Generate Flashcards"):
            prompt = (
                "Create short flashcards from this lecture.\n"
                "Each flashcard should have a question and a short answer."
            )
            st.session_state.flash_text = generate_with_prompt(prompt, st.session_state.transcript)

        if st.session_state.flash_text:
            st.subheader("Generated Flashcards")
            st.text_area("", st.session_state.flash_text, height=350)
