import streamlit as st
import os
import tempfile
import time
import test_rag
from test_rag import database_exists, is_vision_available, query_rag_with_images
from generate_caption import IMAGE_STORAGE_PATH
from PIL import Image
import base64

# Clear cache and force refresh
st.set_page_config(layout="wide")
st.cache_data.clear()
st.cache_resource.clear()

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.last_db_state = False

# Add welcome message if no messages exist
if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload PDFs and images, and I'll answer questions about them."}
    ]

st.markdown("""
<style>
    .image-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .image-caption {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def display_message_with_images(message):
    """Display a message that may contain both text and images"""
    if isinstance(message, dict):
        # This is a message with structured content
        if "text" in message:
            st.write(message["text"])
        
        if "images" in message and message["images"]:
            st.markdown("**Related Images:**")
            
            # Display images in columns
            num_images = len(message["images"])
            cols = st.columns(min(3, num_images))
            
            for idx, img_info in enumerate(message["images"]):
                col_idx = idx % 3
                with cols[col_idx]:
                    try:
                        # Create container for image
                        with st.container():
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            
                            # Display image
                            image = Image.open(img_info["path"])
                            st.image(
                                image, 
                                caption=img_info["filename"],
                                use_container_width=True
                            )
                            
                            # Show caption preview with expander for full caption
                            with st.expander("View Analysis"):
                                st.write(img_info["caption"][:500] + "..." if len(img_info["caption"]) > 500 else img_info["caption"])
                            
                            st.markdown('</div>', unsafe_allow_html=True)

                            
                    except Exception as e:
                        st.error(f"Could not display image: {str(e)}")
    else:
        st.write(message)

# Main chat interface
st.title("Multimodal RAG Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        display_message_with_images(message["content"])

with st.sidebar:
    st.title("File Upload")
    
    # Refresh button
    if st.button("Refresh Page"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    db_exists = test_rag.database_exists()
    vision_available = is_vision_available()
    
    st.write("**System Status:**")
    st.write(f"ChromaDB: {'Ready' if db_exists else 'Not Found'}")
    st.write(f"Vision: {'Available' if vision_available else 'Not Available'}")
    
    if os.path.exists(IMAGE_STORAGE_PATH):
        image_count = len([f for f in os.listdir(IMAGE_STORAGE_PATH) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])
        st.write(f"• Images Stored: {image_count}")

    # Database management section
    st.subheader("Database Controls")

    if st.button("Clear Database"):
        try:
            test_rag.clear_chroma()
            # Also clear image storage
            if os.path.exists(IMAGE_STORAGE_PATH):
                for file in os.listdir(IMAGE_STORAGE_PATH):
                    file_path = os.path.join(IMAGE_STORAGE_PATH, file)
                    try:
                        os.remove(file_path)
                    except:
                        pass
            st.success("Database and images cleared!")
            st.session_state.messages = [
                {"role": "assistant", "content": "Database has been cleared. Please upload new files."}
            ]
            st.session_state.pdf_processed = False
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing database: {e}")

    # File upload section
    st.subheader("Upload PDF Files")
    pdf_docs = st.file_uploader(
        "Choose PDF files", 
        accept_multiple_files=True, 
        type="pdf",
        help="Upload one or more PDF files to create your knowledge base",
        label_visibility="collapsed"
    )

    st.subheader("Upload Images")
    image_files = st.file_uploader(
        "Choose image files", 
        accept_multiple_files=True, 
        type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"],
        help="Upload one or more image files to add to your knowledge base",
        label_visibility="collapsed"
    )

    if st.button("Process Files", type="primary"):
        if pdf_docs or image_files:
            with st.spinner("Processing files..."):
                try:
                    success, num_chunks = test_rag.process_and_save_multimodal(pdf_docs, image_files)
                    
                    if success:
                        file_summary = []
                        if pdf_docs:
                            file_summary.append(f"{len(pdf_docs)} PDFs")
                        if image_files:
                            file_summary.append(f"{len(image_files)} images")
                        
                        st.success(f"Successfully processed {' and '.join(file_summary)} with {num_chunks} total chunks!")
                        
                        # Force clear all caches
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        
                        # Update state
                        st.session_state.pdf_processed = True
                        
                        # Create success message with structure
                        success_content = {
                            "text": f"Ready! I've processed your files ({' and '.join(file_summary)}). Ask me anything about their content!",
                            "images": []
                        }
                        st.session_state.messages = [
                            {"role": "assistant", "content": success_content}
                        ]
                        
                        # Add a small delay before rerun
                        time.sleep(2)
                        st.rerun()
                        
                    else:
                        st.error("Failed to process files. Check terminal for detailed errors.")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
        else:
            st.warning("Please upload PDF files and/or images first.")

    # Debug section
    with st.expander("Debug Info"):
        st.write(f"Messages in session: {len(st.session_state.messages)}")
        st.write(f"PDF processed flag: {st.session_state.pdf_processed}")
        
        if st.button("Clear Session State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Main chat interface
st.subheader("Chat with Your Documents and Images")

# Auto-refresh logic
current_db_state = test_rag.database_exists()
if st.session_state.last_db_state != current_db_state:
    st.session_state.last_db_state = current_db_state
    st.rerun()

if test_rag.database_exists():
    st.success("Database is ready. Ask questions about your uploaded files.")
else:
    st.warning("No database found. Please upload files using the sidebar.")

# Chat input
user_input = st.chat_input("Ask a question about your documents and images...")

if user_input and user_input.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate RAG response with images
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and images..."):
            try:
                if not test_rag.database_exists():
                    st.error("No database found. Please upload files first.")
                    response_data = {
                        "text": "No database available. Please upload documents and images first.",
                        "images": []
                    }
                else:
                    formatted_response, response_text, display_images = query_rag_with_images(user_input)
                    
                    # Create structured response
                    response_data = {
                        "text": response_text,
                        "images": display_images
                    }
                    
                    # Display the response
                    display_message_with_images(response_data)
                    
            except Exception as e:
                error_msg = f"Error querying database: {str(e)}"
                st.error(error_msg)
                response_data = {
                    "text": error_msg,
                    "images": []
                }
                st.write(error_msg)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    # Force rerun to show updated messages
    st.rerun()