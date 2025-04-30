# 🧠 Multimodal Hate Speech Detection App

This is a Streamlit-based application for detecting hate speech across **text, documents, and audio/video files**. It leverages **OpenAI GPT-3.5** for classification and explanation, and **Whisper** for audio transcription. Optional multimodal analysis integrates video frame extraction and image-text alignment.

---

## 🔍 Key Features

- ✍️ **Text Input**: Detect hate or non-hate speech in typed content  
- 📄 **Document Upload**: Analyze hate speech in PDF or Word files  
- 🎧 **Audio/Video Upload**: Automatically transcribe and analyze audio or video content  
- 🌐 **Multilingual Support**: Auto-detects language and works across various languages  
- 🤖 **OpenAI GPT Classification**: Provides hate/non-hate label and short explanation  
- 🔈 **Whisper Transcription**: Converts speech to text for analysis  
- 🖼️ **(Optional) Video Frame Analysis**: Supports image-text multimodal classification if a ResNet model is provided  
- 📥 **Export Support**: Download CSV results of video-text analysis  

---

## 🖥️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure you have `ffmpeg` installed (needed for Whisper/video processing).

### 3. Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI key:

```env
OPENAI_API_KEY=your-openai-key
```

Alternatively, you can hardcode it (as in the script) or modify the script to load from `.env`.

### 4. Run the Application

```bash
streamlit run streamlit_app_final.py
```

---

## 📁 Project Structure

- `streamlit_app_final.py` – Main app logic (text, audio, video handling)  
- `pipeline.py` – (Optional) Utility functions for frame extraction, multimodal analysis  
- `requirements.txt` – Python dependencies  
- `.env` – API key config (optional)  
- `harmful_content_classifier_resnet50.pth` – (Optional) Pretrained image classifier model (if multimodal)  

---

## 🧠 Model Logic

### Text Classification (GPT)
OpenAI GPT-3.5 is used with the following prompt:

```
You are a hate speech detection assistant. Analyze the given text.
First, classify it strictly as either 'hate' or 'non-hate'.
Then provide a brief explanation (1–2 sentences) why it is classified that way.
Format your reply as:
Label: [hate/non-hate]
Explanation: [your reason]
```

### Audio Transcription
Whisper's `"base"` model is used to convert audio/video to text for further analysis.

### Optional: Multimodal Analysis
If enabled, the app extracts frames from uploaded videos and runs image-text analysis (requires a custom model and pipeline).

---

## 📦 Output

- Text and document results are shown on screen with GPT’s explanation  
- Video analysis results are shown as a dataframe and downloadable as CSV