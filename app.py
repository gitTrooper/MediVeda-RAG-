import os
import shutil
import tempfile
import io
import uuid
import time
from pathlib import Path
import gradio as gr
import torch
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import fitz  # PyMuPDF for more robust PDF handling
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Constants
KNOWLEDGE_DIR = "medical_knowledge"
VECTOR_STORE_PATH = "vectorstore"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Gated model requiring authentication
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Create a temporary directory for file uploads
TEMP_DIR = os.path.join(tempfile.gettempdir(), "medical_reports")
os.makedirs(TEMP_DIR, exist_ok=True)

# Dictionary to track temporary files for cleanup
temp_files = {}

# Get HF token from environment variables (set in HF Spaces secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables. You may not be able to access gated models.")

class MedicalReportAnalyzer:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.user_report_data = "No report data available."  # Default value
        # Initialize everything
        self._load_or_create_vector_store()
        self._initialize_llm()
        self._setup_qa_chain()

    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one from knowledge documents"""
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Check if vector store exists
        if os.path.exists(VECTOR_STORE_PATH):
            print("Loading existing vector store...")
            self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        else:
            print("Creating new vector store from documents...")
            # Create knowledge directory if it doesn't exist
            os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
            
            # Check if there are documents to process
            if len(os.listdir(KNOWLEDGE_DIR)) == 0:
                print(f"Warning: No documents found in {KNOWLEDGE_DIR}. Please add medical PDFs.")
                # Initialize empty vector store
                self.vector_store = FAISS.from_texts(["No medical knowledge available yet."], embeddings)
                self.vector_store.save_local(VECTOR_STORE_PATH)
                return
            
            # Load all PDFs from the knowledge directory
            try:
                # First try with DirectoryLoader
                loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create and save the vector store
                self.vector_store = FAISS.from_documents(chunks, embeddings)
                self.vector_store.save_local(VECTOR_STORE_PATH)
            except Exception as e:
                print(f"Error loading documents with DirectoryLoader: {str(e)}")
                # Initialize with minimal data
                self.vector_store = FAISS.from_texts(["No medical knowledge available yet."], embeddings)
                self.vector_store.save_local(VECTOR_STORE_PATH)

    def _initialize_llm(self):
        """Initialize the language model with HF token authentication"""
        print(f"Loading model {MODEL_NAME} on {DEVICE}...")
        try:
            # Use the HF_TOKEN for authentication
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto",
                load_in_8bit=DEVICE == "cuda",  # Use 8-bit quantization if on CUDA
            )
            
            # Create a text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            # Create LangChain wrapper around the pipeline
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Error loading the model: {str(e)}")
            print("Falling back to a non-gated model...")
            # Fallback to a non-gated model
            fallback_model = "google/flan-t5-large"
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto"
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

    def _setup_qa_chain(self):
        """Set up the question-answering chain"""
        # Define a custom prompt template for medical analysis
        template = """
        You are a medical assistant analyzing patient medical reports. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Patient Report Summary: {patient_data}

        Context from medical knowledge base: {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "patient_data"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

    def extract_text_from_pdf_pymupdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF (more robust than PyPDF)"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"PyMuPDF extraction error: {str(e)}")
            return None

    def extract_text_from_pdf_pypdf(self, pdf_path):
        """Extract text using PyPDF as a backup method"""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            return "\n".join([page.page_content for page in pages])
        except Exception as e:
            print(f"PyPDF extraction error: {str(e)}")
            return None

    def process_user_report(self, report_file):
        """Process the uploaded medical report with multiple fallback methods"""
        if report_file is None:
            return "No file uploaded. Please upload a medical report."
        
        # Ensure the uploaded file is read as bytes
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy the uploaded file to the temp directory
            temp_file_path = os.path.join(temp_dir, "user_report.pdf")
            
            # Handle file based on its type
            try:
                if isinstance(report_file, str):  # If it's a file path
                    shutil.copy(report_file, temp_file_path)
                elif hasattr(report_file, 'name'):  # Gradio file object
                    with open(temp_file_path, 'wb') as f:
                        with open(report_file.name, 'rb') as source:
                            f.write(source.read())
                else:  # Try to handle as bytes or file-like object
                    with open(temp_file_path, 'wb') as f:
                        f.write(report_file.read() if hasattr(report_file, 'read') else report_file)
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                return f"Error saving the uploaded file: {str(e)}"
                
            # Try multiple methods to extract text from the PDF
            text = None
            
            # Method 1: PyMuPDF
            text = self.extract_text_from_pdf_pymupdf(temp_file_path)
            
            # Method 2: PyPDF as fallback
            if not text:
                text = self.extract_text_from_pdf_pypdf(temp_file_path)
                
            # Method 3: Last resort - try to read as raw text
            if not text:
                try:
                    with open(temp_file_path, 'r', errors='ignore') as f:
                        text = f.read()
                except Exception as e:
                    print(f"Raw text reading error: {str(e)}")
            
            # If we got text, process it
            if text and len(text.strip()) > 0:
                # Store the text
                self.user_report_data = text
                
                # Split into chunks if needed
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                return f"Report processed successfully. Extracted approximately {len(chunks)} text chunks."
            else:
                self.user_report_data = "Unable to extract text from the provided PDF. This is an empty report placeholder."
                return "Warning: Could not extract text from the PDF. The file may be corrupted, password-protected, or contain only images. Processing will continue with limited data."
                
        finally:
            # Clean up the temporary directory and file
            shutil.rmtree(temp_dir)

    def answer_question(self, question):
        """Answer a question based on the uploaded report and knowledge base"""
        if not self.user_report_data or self.user_report_data == "No report data available.":
            return "No report has been processed or text extraction failed. Please upload a medical report first."
        
        # Get context from knowledge base
        try:
            retrieved_docs = self.vector_store.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create the inputs dict for the QA chain
            inputs = {
                "query": question,
                "context": context,
                "patient_data": self.user_report_data
            }
            
            # Run the chain with the correct parameter structure
            result = self.qa_chain(inputs)
            
            # Extract the answer from the result
            if isinstance(result, dict) and 'result' in result:
                return result['result']
            else:
                return str(result)
                
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            error_msg = f"Error processing your question: {str(e)}."
            
            # Try direct LLM call as fallback
            try:
                direct_prompt = f"""
                Question about medical report: {question}
                
                Patient data available: {self.user_report_data[:500]}... (truncated)
                
                Please answer based on this information:
                """
                
                direct_result = self.llm(direct_prompt)
                return f"{error_msg} Fallback answer: {direct_result}"
            except:
                return f"{error_msg} Please try a different question or report."

# Initialize the analyzer outside the Gradio interface
analyzer = MedicalReportAnalyzer()

# Define the Gradio interface
with gr.Blocks(title="Medical Report Analyzer") as demo:
    gr.Markdown("# Medical Report Analyzer")
    gr.Markdown("Upload your medical report and ask questions about it. The system will analyze your report and provide answers based on medical knowledge.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload component
            report_file = gr.File(label="Upload Medical Report (PDF)")
            upload_button = gr.Button("Process Report")
            upload_output = gr.Textbox(label="Processing Status")
            
        with gr.Column(scale=2):
            # Q&A interface
            question_input = gr.Textbox(label="Ask a question about your report")
            answer_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer")
    
    # Set up the event handlers
    upload_button.click(
        fn=analyzer.process_user_report,
        inputs=[report_file],
        outputs=[upload_output]
    )
    
    answer_button.click(
        fn=analyzer.answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the Gradio app
app = gr.mount_gradio_app(app, demo, path="/gradio")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file temporarily and return its URL.
    File will be automatically deleted after processing or via cleanup endpoint.
    """
    try:
        # Create a unique filename to avoid collisions
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_filename = f"{file_id}{file_extension}"
        file_path = os.path.join(TEMP_DIR, temp_filename)
        
        # Save file to temporary directory
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Store file info for cleanup
        temp_files[file_path] = {
            "original_name": file.filename,
            "created_at": os.path.getctime(file_path)
        }
        
        # Return file URL that can be used by the model
        file_url = f"/tmp/medical_reports/{temp_filename}"
        
        return JSONResponse({
            "success": True,
            "url": file_url,
            "filename": file.filename
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.delete("/cleanup")
async def cleanup_file(request: Request):
    """
    Delete a temporary file by its URL.
    """
    params = dict(request.query_params)
    file_url = params.get("file")
    
    if not file_url:
        return JSONResponse({
            "success": False,
            "error": "No file URL provided"
        }, status_code=400)
    
    try:
        # Extract filename from URL
        filename = os.path.basename(file_url)
        file_path = os.path.join(TEMP_DIR, filename)
        
        # Check if file exists
        if os.path.exists(file_path):
            os.remove(file_path)
            if file_path in temp_files:
                del temp_files[file_path]
                
            return JSONResponse({
                "success": True,
                "message": "File deleted successfully"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "File not found"
            }, status_code=404)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# Cleanup old files periodically (optional)
@app.on_event("startup")
async def startup_cleanup():
    """
    Clean up any old temporary files at startup.
    """
    try:
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > max_age:
                    os.remove(file_path)
    except Exception as e:
        print(f"Error during startup cleanup: {e}")

# This makes it run when directly executed (not when imported)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)