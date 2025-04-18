import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from collections import OrderedDict

# Retrieve HF_TOKEN from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

# Constants
DATA_PATH = "dataFolder/"
DB_FAISS_PATH = "/tmp/vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"

# Cache directory
CACHE_DIR = "/tmp/models_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="rishi002/all-MiniLM-L6-v2",
    cache_folder=CACHE_DIR
)

# Load or create FAISS database
def load_or_create_faiss():
    if not os.path.exists(DB_FAISS_PATH):
        print("🔄 Creating FAISS Database...")
        from embeddings import load_pdf_files, create_chunks  # Import functions from embeddings.py

        documents = load_pdf_files(DATA_PATH)  # Load PDFs
        text_chunks = create_chunks(documents)  # Split into Chunks

        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
    else:
        print("✅ FAISS Database Exists. Loading...")

    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load FAISS database
db = load_or_create_faiss()

# Load LLM
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Only provide information from the given context.
Keep your answer concise and avoid repeating the same information.
Each important point should be stated only once. 
NOTE: SUMMARIZE YOUR ANSWERS STRICTLY WITHIN 300 WORDS.

Context: {context}
Question: {question}

Start the answer directly.
"""

# Create the QA chain
def create_qa_chain():
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

qa_chain = create_qa_chain()

# Define Gradio Interface
def ask_question(query: str):
    try:
        response = qa_chain.invoke({'query': query})
        
        # Get the raw result
        result = response["result"]
        
        # Remove duplicates by splitting into sentences and keeping only unique ones
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        # Use OrderedDict to preserve order while removing duplicates
        unique_sentences = list(OrderedDict.fromkeys(sentences))
        
        # Rejoin with periods
        cleaned_result = '. '.join(unique_sentences) + '.'
        
        # Limit length if needed
        if len(cleaned_result) > 500:
            cleaned_result = cleaned_result[:500] + "..."
        
        return cleaned_result, [doc.metadata for doc in response["source_documents"]]
    except Exception as e:
        return f"Error: {str(e)}", []

# Create Gradio interface
iface = gr.Interface(fn=ask_question, inputs="text", outputs=["text", "json"])

# Launch the Gradio app (this will auto-host on Hugging Face Space)
iface.launch(share=True)
