from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile
import shutil
import time
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")


CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_rag_db")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

Answer the question based on the above context: {question}
"""

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=100,
        length_function=len, 
        add_start_index=True, 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def process_pdf_files(pdf_files):
    all_chunks = []
    
    for pdf_file in pdf_files:
        # For Streamlit file uploader objects
        if hasattr(pdf_file, 'name') and hasattr(pdf_file, 'getvalue'):
            # This is a Streamlit UploadedFile object
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                chunks = split_text(documents)
                all_chunks.extend(chunks)
                print(f"Processed: {pdf_file.name} → {len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        # For file paths (string)
        elif isinstance(pdf_file, str) and os.path.exists(pdf_file):
            try:
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                chunks = split_text(documents)
                all_chunks.extend(chunks)
                print(f"Processed: {pdf_file} → {len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to process {pdf_file}: {str(e)}")
    
    return all_chunks

def save_to_chroma(chunks: list[Document]):
    print("Creating ChromaDB index with persistence...")
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:        
            # Clear existing database first to avoid conflicts
            if os.path.exists(CHROMA_PATH):
                print(f"Attempt {attempt + 1}: Clearing existing ChromaDB...")
                try:
                    shutil.rmtree(CHROMA_PATH)
                    print("Existing database cleared")
                except PermissionError as e:
                    print(f"Permission error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        print(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return save_to_chroma_force_clear(chunks)
            
            embedding_function = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION, 
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY
            )
            
            print("Embedding function created successfully")
            
            # Create ChromaDB with persistence
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=CHROMA_PATH
            )
            
            # Explicitly persist to disk
            db.persist()
            print("Database persisted successfully")
            
            # Verify creation
            if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
                files = os.listdir(CHROMA_PATH)
                print(f"SUCCESS: Saved ChromaDB with {len(chunks)} chunks")
                print(f"Database path: {CHROMA_PATH}")
                print(f"Files created: {len(files)}")
                return True  
            else:
                print("FAILED: ChromaDB files were not created!")
                return False  
                
        except Exception as e:
            print(f"ERROR creating ChromaDB (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                import traceback
                traceback.print_exc()
                return False

def save_to_chroma_force_clear(chunks: list[Document]):
    print("Using force clear method...")
    
    try:
        # Try to close any open Chroma connections first
        import gc
        gc.collect()
        
        # Use different deletion strategy
        if os.path.exists(CHROMA_PATH):
            for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    try:
                        os.chmod(file_path, 0o777)  # Change permissions
                        os.unlink(file_path)  # Delete file
                    except Exception as e:
                        print(f"Could not delete {file_path}: {e}")
                
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        os.rmdir(dir_path)
                    except Exception as e:
                        print(f"Could not remove directory {dir_path}: {e}")
            
            # Final cleanup
            try:
                shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            except:
                pass
        
        # Wait a bit
        time.sleep(3)
        
        # Now create new database
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=CHROMA_PATH
        )
        
        db.persist()
        print("Force clear method successful")
        return True
        
    except Exception as e:
        print(f"Force clear also failed: {e}")
        return False

def save_to_chroma_fallback(chunks: list[Document]):
    print("Trying fallback ChromaDB creation...")
    
    try:
        # Try in-memory first, then persist
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        # Create in temporary directory first
        temp_chroma_path = os.path.join(tempfile.gettempdir(), "temp_chroma_db")
        
        if os.path.exists(temp_chroma_path):
            shutil.rmtree(temp_chroma_path)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=temp_chroma_path
        )
        
        db.persist()
        
        # Now copy to final location
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        shutil.copytree(temp_chroma_path, CHROMA_PATH)
        
        # Clean up temp
        shutil.rmtree(temp_chroma_path)
        
        print("Fallback database creation successful")
        return True
        
    except Exception as e:
        print(f"Fallback also failed: {e}")
        return False

def process_and_save_pdfs(pdf_files):
    print(f"Processing {len(pdf_files)} PDF files...")
    chunks = process_pdf_files(pdf_files)
    
    if chunks:
        success = save_to_chroma(chunks)
        return success, len(chunks)
    else:
        return False, 0

def load_chroma():
    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION, 
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    
    # Load from disk with error handling
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
        print("ChromaDB loaded successfully")
        return db
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None

def query_rag(query_text: str):
    # Check if ChromaDB exists
    if not database_exists():
        error_msg = "No database found. Please upload and process PDF files first."
        return error_msg, error_msg
    
    try:
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
        
        # Search documents
        results = db.similarity_search(query_text, k=3)
        
        if not results:
            error_msg = "No relevant information found in the documents."
            return error_msg, error_msg
        
        # Combine context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        
        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Initialize OpenAI chat model
        model = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )

        # Generate response
        response = model.invoke(prompt)
        response_text = response.content
        
        # Get sources
        sources = [doc.metadata.get("source", "Unknown") for doc in results]
        
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response, response_text
        
    except Exception as e:
        error_msg = f"Error querying database: {str(e)}"
        print(f"Query error: {error_msg}")
        return error_msg, error_msg

def clear_chroma():
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Cleared ChromaDB at {CHROMA_PATH}")
            return True
        else:
            print("No ChromaDB to clear")
            return True
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")
        return False

# Check if database exists
def database_exists():
    exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    print(f"Database exists: {exists}")
    return exists