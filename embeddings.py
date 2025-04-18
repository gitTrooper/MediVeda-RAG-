import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Set Paths
DATA_PATH = "dataFolder/"
DB_FAISS_PATH = "/tmp/vectorstore/db_faiss"

# Hugging Face Credentials
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Step 1: Load PDF Files
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Step 2: Create Chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Step 3: Generate Embeddings
def get_embedding_model():
    CACHE_DIR = "/tmp/models_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    embedding_model = HuggingFaceEmbeddings(
        model_name="rishi002/all-MiniLM-L6-v2",
        cache_folder="/tmp/models_cache"
    )
    return embedding_model

# Step 4: Store Embeddings in FAISS
def store_embeddings(text_chunks, embedding_model, db_path):
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    return db

# Step 5: Load FAISS Database
def load_faiss_db(db_path, embedding_model):
    return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# Step 6: Load LLM Model
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

# Step 7: Set Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the user's question.
If the answer is unknown, say you don't know. Do not make up information.
Only respond based on the context.

Context: {context}
Question: {question}

Start your answer directly.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 8: Create Retrieval QA Chain
def create_qa_chain(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

# Create and load all models and FAISS (for Gradio)
def prepare_qa_system():
    # Load and process PDFs, create FAISS index, etc.
    print("🔄 Loading PDFs...")
    documents = load_pdf_files(DATA_PATH)
    
    print("📄 Creating Chunks...")
    text_chunks = create_chunks(documents)
    
    print("🧠 Generating Embeddings...")
    embedding_model = get_embedding_model()
    
    print("💾 Storing in FAISS...")
    db = store_embeddings(text_chunks, embedding_model, DB_FAISS_PATH)

    print("🔄 Loading FAISS Database...")
    db = load_faiss_db(DB_FAISS_PATH, embedding_model)

    print("🤖 Loading LLM...")
    llm = load_llm(HUGGINGFACE_REPO_ID)

    print("🔗 Creating QA Chain...")
    qa_chain = create_qa_chain(llm, db)

    return qa_chain

# Create the QA system and get the chain ready
qa_chain = prepare_qa_system()

# Gradio Interface function
def ask_question(query: str):
    try:
        response = qa_chain.invoke({"query": query})
        return response["result"], [doc.metadata for doc in response["source_documents"]]
    except Exception as e:
        return f"Error: {str(e)}", []

