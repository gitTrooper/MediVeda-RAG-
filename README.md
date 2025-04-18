<div align="center">

# 🩺 MediVeda – AI-Powered Medical Assistant 💬📄

A smart healthcare assistant that answers medical questions using trusted documents and even analyzes your uploaded medical reports. Built with 💖 on **LangChain, HuggingFace, and Gradio**, MediVeda brings the power of modern AI to your fingertips.

![MediVeda Banner](https://imgur.com/your-banner-image.png) <!-- Optional -->

</div>

---

## 🌟 Overview

**MediVeda** is an AI-first medical solution with two core features:

1. 🧠 **Medical QA Chatbot** – Ask any health-related question, and get AI-verified answers from a medical knowledge base.
2. 📄 **Medical Report Analyzer** – Upload your medical PDF reports, and ask patient-specific questions with contextual answers.

Hosted live on **Hugging Face Spaces** and powered by **Phi-3 Mini** and **LangChain’s RetrievalQA**, MediVeda is built to enhance healthcare understanding for everyone.

---

## 🧠 App 1: Medical QA Chatbot (RAG-based)

A fast, document-aware medical chatbot that uses:

| 🔧 Component            | 🧾 Description |
|------------------------|----------------|
| `app.py`               | Backend code for chatbot |
| `FAISS`                | Vector store to store medical document embeddings |
| `LangChain`            | RetrievalQA chain with custom prompt |
| `HuggingFaceEndpoint`  | Runs the `microsoft/Phi-3-mini-4k-instruct` model |
| `Gradio`               | Chat interface |

### ✨ Features
- 🔍 Search over pre-loaded medical PDFs
- 🧠 Uses `MiniLM` embeddings for semantic understanding
- ✂️ Removes repeated info from answers
- 💬 Clean and simple Gradio chat interface
- 🌐 Hosted live on Hugging Face Spaces

---

## 📄 App 2: Medical Report Analyzer (PDF Upload)

This app lets users upload **their own medical PDFs** (like blood reports), extracts the content, and answers questions by merging patient data + external knowledge.

| ⚙️ Component              | 🔍 Description |
|--------------------------|----------------|
| `MedicalReportAnalyzer`  | Core class for report handling and QA |
| `PyMuPDF` + `PyPDF`      | Text extraction (multi-layer fallback) |
| `LangChain`              | QA pipeline with `patient_data` + `context` |
| `FAISS`                  | Medical corpus index |
| `HuggingFacePipeline`    | Direct Phi-3 or Flan-T5 generation |
| `Gradio Blocks` UI       | Upload + question form |

### ✨ Features
- 📁 Upload PDF reports
- 🔎 Extracts and splits report text
- 🧬 Merges personal data with medical document context
- ❌ Robust fallback methods if PDF parsing fails
- 🤖 Answers personalized health questions

---

## 🗂️ Project Structure

├── app.py # Launch script (Gradio) 
├── embeddings.py # (If using for PDF chunking in chatbot) 
├── dataFolder/ # PDF documents for chatbot context 
├── ReportAnalysis/ # Folder with medical report analyzer app 
│── app.py # Report analyzer with upload and QA 
├── vectorstore/ # FAISS vector index storage 
├── requirements.txt # All dependencies


💬 Example Use Cases
🧠 Chatbot:
"What are the symptoms of asthma?"
"Can I take paracetamol with ibuprofen?"

📄 Report Analyzer:
Upload: Blood Report
Ask: "Is my HbA1c too high?"
Ask: "What does low hemoglobin mean in this report?"

📦 Requirements

gradio
langchain
transformers
sentence-transformers
PyMuPDF
PyPDF2
faiss-cpu
torch

🔮 Roadmap

 - ✅ Dual app integration (Chatbot + Report Analyzer)

 - 🔐 GPT-4 Health verification (future)

 - 📱 Android app with integrated Hugging Face APIs

 - 📊 Diet planning and exercise suggestions

 - 📷 OCR support for scanned image reports



---

## 💻 Local Setup

```bash
# Clone the repo
git clone https://github.com/gitTrooper/MediVeda-RAG-.git
cd MediVeda-RAG-

# Install dependencies
pip install -r requirements.txt

# Set your Hugging Face API token
export HF_TOKEN=your_token_here

# Run the chatbot
python app.py

# OR run the report analyzer (inside ReportAnalysis)
cd ReportAnalysis
python app.py


