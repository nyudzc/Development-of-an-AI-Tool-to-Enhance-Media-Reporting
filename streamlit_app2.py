# ✅ Full Streamlit App: text + PDF + audio/video integration using pipeline.py

import streamlit as st
import tempfile
import os
import pandas as pd
from streamlit.components.v1 import html
from pipeline import (
    extract_frames, transcribe_audio, load_text_mapping,
    run_multimodal_analysis
)

st.set_page_config(page_title="Multilingual Hate Speech Detector", layout="wide")
st.title("💬 Multilingual Hate Speech Detector (Text + Audio/Video)")

# Input mode selection
input_mode = st.radio("Select input type:", ["✍️ Text Input", "📄 Upload PDF / Word Files", "🎤 Upload Audio/Video"])

# === TEXT INPUT ===
if input_mode == "✍️ Text Input":
    text_input = st.text_area("Enter your text here:", height=200)
    if st.button("Analyze Text"):
        st.write("⚠️ Placeholder: integrate your text->Dify API call here.")
        # Show fake result
        st.success("This text is non-hate (mock).")

# === DOCUMENT UPLOAD ===
elif input_mode == "📄 Upload PDF / Word Files":
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        st.write("⚠️ Placeholder: implement PDF/DOCX text extraction and analysis.")
        for file in uploaded_files:
            st.success(f"Processed file: {file.name}")

# === AUDIO/VIDEO UPLOAD ===
elif input_mode == "🎤 Upload Audio/Video":
    uploaded_video = st.file_uploader("Upload an audio or video file", type=["mp3", "wav", "mp4", "mov"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_video.name) as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        st.success("✅ File uploaded. Starting analysis...")

        # Set paths
        frame_folder = tempfile.mkdtemp(prefix="frames_")
        transcript_csv = os.path.join(tempfile.gettempdir(), "transcripts.csv")
        model_path = "harmful_content_classifier_resnet50.pth"  # Make sure this file exists
        class_labels = ['harmful', 'xenophobic', 'misinformation', 'neutral']

        try:
            st.info("📸 Extracting frames...")
            extract_frames(video_path, frame_folder)

            st.info("🎧 Transcribing audio...")
            transcribe_audio(video_path, transcript_csv)

            st.info("🧠 Running multimodal hate speech analysis...")
            text_mapping = load_text_mapping(transcript_csv)
            df = run_multimodal_analysis(frame_folder, text_mapping, model_path, class_labels)

            st.success("✅ Analysis complete.")
            st.write("🧾 First 5 Results:")
            st.dataframe(df.head())

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full CSV", data=csv_data, file_name="multimodal_results.csv")
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
