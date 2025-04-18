<div align="center">

# 🩺 MediVeda – Your Personalized Medical AI Assistant 🤖💊

</div>

---

## 🌟 Overview

**MediVeda** is an AI-powered medical chatbot designed to assist users with symptom analysis, medical queries, and health-related insights using real-time document retrieval and cutting-edge large language models. 🧠📄

Powered by [🧬 Phi-3 Mini (by Microsoft)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [FAISS](https://github.com/facebookresearch/faiss) vector database, and [LangChain](https://www.langchain.com/), MediVeda is hosted seamlessly on [🤗 Hugging Face Spaces](https://huggingface.co/spaces).

---

## ✨ Features

| 💡 Feature                      | 🔍 Description |
|-------------------------------|----------------|
| 📁 PDF-based Medical Context   | Ingests and indexes medical documents using FAISS for fast retrieval. |
| 🧠 LLM-backed QA System        | Uses Phi-3 Mini LLM for intelligent, context-aware medical responses. |
| 🧾 Custom Prompt Engineering   | Ensures concise and accurate medical summaries with a 300-word limit. |
| 💬 Gradio UI                   | Clean and interactive chat interface for users to ask health-related questions. |
| 🔐 Secure LLM Access           | Retrieves Hugging Face token securely using environment variables. |
| ♻️ Duplicate-Free Responses    | Filters out repeated sentences to maintain clarity in answers. |
| 🚀 Hugging Face Deployment     | Deployed on Hugging Face Spaces with public sharing enabled. |

---

## 🧠 Tech Stack

- **Frontend/UI**: [Gradio](https://www.gradio.app/)
- **LLM**: [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Embeddings**: `rishi002/all-MiniLM-L6-v2` (via HuggingFaceEmbeddings)
- **Retrieval**: [FAISS Vector Store](https://github.com/facebookresearch/faiss)
- **Frameworks**: [LangChain](https://www.langchain.com/), [Hugging Face](https://huggingface.co/)

---

⚠️ Make sure you add your medical PDFs to the dataFolder/ directory before running the app.

🚀 How It Works
 - PDF Ingestion: Loads and chunks PDFs into small text pieces.
  
 - Embedding Generation: Each chunk is converted to vector embeddings using MiniLM.
  
 - FAISS Vector Store: Stores the chunks in a searchable FAISS database.
  
 - Prompt Engineering: Custom prompt used to generate concise, relevant answers.
  
 - LLM Response: Query is answered using context retrieved from documents and Phi-3 Mini.
  
 - UI Display: Result is shown via Gradio, with source metadata included.

🛠️ Setup Instructions

  - Clone the Repo
      git clone https://github.com/your-username/mediveda.git
      cd mediveda
  
  - Install Requirements
      pip install -r requirements.txt
    
  - Set Hugging Face Token
      export HF_TOKEN=your_huggingface_token
    
  - Run the App
      python app.py

## 📂 Project Structure

```bash
.
├── app.py                 # Main application logic
├── dataFolder/           # Folder to place all medical PDFs
├── embeddings.py         # Functions for loading PDFs and creating chunks (assumed external)
└── /tmp/vectorstore/     # Temporary storage for FAISS vector store


