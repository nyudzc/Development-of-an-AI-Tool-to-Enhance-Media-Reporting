import streamlit as st
import tempfile
import os
import pandas as pd
import whisper
from openai import OpenAI
from pipeline import extract_frames, transcribe_audio, load_text_mapping, run_multimodal_analysis
from sklearn.metrics import classification_report, confusion_matrix

# Initialize OpenAI client
client = OpenAI(api_key="Your-API-Key")  # <-- Replace here with your OpenAI Key

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Function to detect hate speech and explanation
def detect_hate_openai(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hate speech detection assistant. Analyze the given text."
                        " First, classify it strictly as either 'hate' or 'non-hate'."
                        " Then provide a brief explanation (1-2 sentences) why it is classified that way."
                        " Format your reply as:\nLabel: [hate/non-hate]\nExplanation: [your reason]"
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {str(e)}"

# Streamlit UI setup
st.set_page_config(page_title="Multimodal Hate Speech Detector", layout="wide")
st.title("💬 Multimodal Hate Speech Detector")

input_mode = st.radio("Select input type:", ["✍️ Text Input", "📄 Upload PDF/Word", "🎤 Upload Audio/Video"])

# === Text Input Mode ===
if input_mode == "✍️ Text Input":
    text_input = st.text_area("Enter your text below:", height=200)
    if st.button("Analyze Text"):
        if text_input.strip() != "":
            with st.spinner("Analyzing with OpenAI..."):
                prediction = detect_hate_openai(text_input)
            st.success(prediction)
        else:
            st.warning("Please enter some text.")

# === PDF/Word Upload Mode ===
elif input_mode == "📄 Upload PDF/Word":
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        extracted_texts = []

        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split(".")[-1].lower()
            file_bytes = uploaded_file.read()

            if file_type == "pdf":
                import fitz  # PyMuPDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    pdf_doc = fitz.open(tmp_file.name)
                    text = ""
                    for page in pdf_doc:
                        text += page.get_text()
                    extracted_texts.append(text)

            elif file_type == "docx":
                import docx
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                    tmp_file.write(file_bytes)
                    doc = docx.Document(tmp_file.name)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    extracted_texts.append(text)

        if extracted_texts:
            for idx, text in enumerate(extracted_texts):
                st.subheader(f"Document {idx + 1} Analysis Result:")
                with st.spinner("Analyzing extracted text..."):
                    prediction = detect_hate_openai(text)
                st.success(prediction)

# === Audio/Video Upload Mode ===
elif input_mode == "🎤 Upload Audio/Video":
    uploaded_media = st.file_uploader("Upload an audio or video file", type=["mp3", "wav", "mp4", "mov"])

    if uploaded_media:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_media.name) as tmp_file:
            tmp_file.write(uploaded_media.read())
            media_path = tmp_file.name

        st.success("File uploaded successfully. Starting analysis...")

        ext = os.path.splitext(uploaded_media.name)[-1].lower()
        transcripts = []

        # Transcribe audio
        try:
            st.info("Transcribing audio...")
            result = whisper_model.transcribe(media_path)
            transcript_text = result["text"]
            transcripts.append(transcript_text)
            st.success("Audio transcription completed.")
        except Exception as e:
            st.error(f"Audio transcription failed: {str(e)}")

        # Extract frames and run analysis if video
        if ext in [".mp4", ".mov", ".avi", ".mkv"]:
            try:
                frame_folder = tempfile.mkdtemp(prefix="frames_")
                model_path = "harmful_content_classifier_resnet50.pth"  # Optional (if you have a model)
                class_labels = ['harmful', 'xenophobic', 'misinformation', 'neutral']

                st.info("Extracting frames...")
                extract_frames(media_path, frame_folder)

                st.info("Running multimodal analysis...")
                text_mapping = {f"frame_{i}": transcripts[0] for i in range(1000)}
                df = run_multimodal_analysis(frame_folder, text_mapping, model_path, class_labels)

                st.success("Multimodal analysis completed.")
                st.dataframe(df.head())

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results CSV", data=csv_data, file_name="multimodal_results.csv")
            except Exception as e:
                st.error(f"Multimodal analysis failed: {str(e)}")

        # Analyze final transcribed text
        if transcripts:
            st.info("Analyzing transcribed text with OpenAI...")
            final_text = transcripts[0]
            with st.spinner("Getting prediction and explanation..."):
                prediction = detect_hate_openai(final_text)
            st.success(prediction)
