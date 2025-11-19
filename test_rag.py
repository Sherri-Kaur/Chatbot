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
from generate_caption import generate_image_captions, is_vision_available

load_dotenv()

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")
CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_rag_db")

PROMPT_TEMPLATE = """
Answer the question based on the following context which may include text from PDFs and descriptions of images:

{context}

Question: {question}

If the context includes image descriptions, you can refer to visual content when relevant.
Provide a comprehensive answer drawing from both text and visual information when available.
"""

def validate_documents(documents: list[Document]):
    """Validate and clean documents before processing"""
    valid_documents = []
    for doc in documents:
        if not hasattr(doc, 'page_content') or doc.page_content is None:
            continue
        cleaned_content = str(doc.page_content).strip()
        if len(cleaned_content) < 10:
            continue
        valid_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata.copy() if doc.metadata else {}
        )
        valid_documents.append(valid_doc)
    print(f"Validated {len(documents)} documents, kept {len(valid_documents)}")
    return valid_documents

def split_text(documents: list[Document]):
    valid_documents = validate_documents(documents)
    if not valid_documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(valid_documents)
    print(f"Split {len(valid_documents)} documents into {len(chunks)} chunks.")
    return chunks

def process_pdf_files(pdf_files):
    all_chunks = []
    for pdf_file in pdf_files:
        if hasattr(pdf_file, 'name') and hasattr(pdf_file, 'getvalue'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                for doc in documents:
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["type"] = "pdf"
                
                chunks = split_text(documents)
                all_chunks.extend(chunks)
                print(f"Processed: {pdf_file.name} → {len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {str(e)}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    return all_chunks

def process_multimodal_files(pdf_files, image_files):
    """Process both PDFs and images for multimodal RAG"""
    all_chunks = []
    
    if pdf_files:
        pdf_chunks = process_pdf_files(pdf_files)
        all_chunks.extend(pdf_chunks)
        print(f"Added {len(pdf_chunks)} PDF chunks")
    
    if image_files:
        image_docs = generate_image_captions(image_files)
        valid_image_docs = validate_documents(image_docs)
        all_chunks.extend(valid_image_docs)
        print(f"Added {len(valid_image_docs)} image caption chunks")
    
    all_chunks = validate_documents(all_chunks)
    print(f"Total valid chunks after processing: {len(all_chunks)}")
    return all_chunks

def save_to_chroma(chunks: list[Document]):
    """Save documents to ChromaDB with retry logic"""
    valid_chunks = validate_documents(chunks)
    if not valid_chunks:
        return False
    
    max_retries, retry_delay = 3, 2
    
    for attempt in range(max_retries):
        try:        
            if os.path.exists(CHROMA_PATH):
                print(f"Attempt {attempt + 1}: Clearing existing ChromaDB...")
                try:
                    shutil.rmtree(CHROMA_PATH)
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return save_to_chroma_force_clear(valid_chunks)
            
            embedding_function = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION, 
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY
            )
            
            db = Chroma.from_documents(
                documents=valid_chunks,
                embedding=embedding_function,
                persist_directory=CHROMA_PATH
            )
            db.persist()
            
            if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
                print(f"SUCCESS: Saved ChromaDB with {len(valid_chunks)} chunks")
                return True
            return False
                
        except Exception as e:
            print(f"ERROR creating ChromaDB (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                import traceback
                traceback.print_exc()
                return False

def save_to_chroma_force_clear(chunks: list[Document]):
    """Force clear and recreate ChromaDB"""
    print("Using force clear method...")
    try:
        import gc
        gc.collect()
        
        if os.path.exists(CHROMA_PATH):
            for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                for name in files:
                    try:
                        os.chmod(os.path.join(root, name), 0o777)
                        os.unlink(os.path.join(root, name))
                    except: pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except: pass
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        
        time.sleep(3)
        
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
        print(f"Force clear failed: {e}")
        return False

def process_and_save_multimodal(pdf_files, image_files):
    """Process both PDFs and images and save to ChromaDB"""
    print(f"Processing {len(pdf_files or [])} PDFs and {len(image_files or [])} images...")
    chunks = process_multimodal_files(pdf_files, image_files)
    return (save_to_chroma(chunks), len(chunks)) if chunks else (False, 0)

def load_chroma():
    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION, 
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    try:
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None

def query_rag(query_text: str):
    if not database_exists():
        return "No database found. Please upload and process files first.", "No database found."
    
    try:
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search(query_text, k=5)
        
        if not results:
            return "No relevant information found.", "No relevant information found."
        
        context_parts = []
        for doc in results:
            source_type = doc.metadata.get("type", "text")
            source_name = doc.metadata.get("source", "Unknown")
            prefix = "[IMAGE DESCRIPTION" if doc.metadata.get("content_type") == "image_caption" or source_type == "image" else "[TEXT DOCUMENT"
            context_parts.append(f"{prefix} - {source_name}]:\n{doc.page_content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)
        
        model = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )

        response = model.invoke(prompt)
        response_text = response.content
        
        sources = [f"{doc.metadata.get('source', 'Unknown')} ({doc.metadata.get('type', 'text')})" for doc in results]
        return f"Response: {response_text}\n\nSources: {', '.join(sources)}", response_text
        
    except Exception as e:
        error_msg = f"Error querying database: {str(e)}"
        return error_msg, error_msg

def query_rag_with_images(query_text: str):
    """Enhanced query function that returns both text and image results"""
    if not database_exists():
        return "No database found. Please upload and process files first.", "No database found.", []
    
    try:
        if not query_text or not query_text.strip():
            return "Please provide a valid question.", "Please provide a valid question.", []
        
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search(query_text, k=5)
        
        if not results:
            return "No relevant information found.", "No relevant information found.", []
        
        image_results = [doc for doc in results if doc.metadata.get("type") == "image" or doc.metadata.get("content_type") == "image_caption"]
        text_results = [doc for doc in results if doc not in image_results]
        
        context_parts = []
        for doc in text_results + image_results:
            source_type = doc.metadata.get("type", "text")
            source_name = doc.metadata.get("source", "Unknown")
            prefix = "[IMAGE DESCRIPTION" if doc.metadata.get("content_type") == "image_caption" or source_type == "image" else "[TEXT DOCUMENT"
            context_parts.append(f"{prefix} - {source_name}]:\n{doc.page_content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)
        
        model = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION, 
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )

        response = model.invoke(prompt)
        response_text = response.content
        
        sources = [f"{doc.metadata.get('source', 'Unknown')} ({doc.metadata.get('type', 'text')})" for doc in results]
        
        display_images = []
        for img_doc in image_results:
            image_info = {
                "path": img_doc.metadata.get("image_path"),
                "filename": img_doc.metadata.get("original_filename", "Unknown"),
                "caption": img_doc.page_content,
                "source": img_doc.metadata.get("source", "Unknown")
            }
            if image_info["path"] and os.path.exists(image_info["path"]):
                display_images.append(image_info)
        
        return f"Response: {response_text}\n\nSources: {', '.join(sources)}", response_text, display_images
        
    except Exception as e:
        error_msg = f"Error querying database: {str(e)}"
        return error_msg, error_msg, []

def clear_chroma():
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Cleared ChromaDB at {CHROMA_PATH}")
            return True
        print("No ChromaDB to clear")
        return True
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")
        return False

def database_exists():
    exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    print(f"Database exists: {exists}")
    return exists

def get_database_stats():
    if not database_exists():
        return {"total_chunks": 0, "pdf_chunks": 0, "image_chunks": 0}
    
    try:
        db = load_chroma()
        if db and hasattr(db._collection, 'count'):
            return {
                "total_chunks": db._collection.count(),
                "pdf_chunks": "N/A",
                "image_chunks": "N/A"
            }
    except:
        pass
    
    return {"total_chunks": "Unknown", "pdf_chunks": "Unknown", "image_chunks": "Unknown"}