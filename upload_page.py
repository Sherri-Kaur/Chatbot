import streamlit as st
import os
import tempfile
import test_rag
from test_rag import database_exists

# Main chat interface
st.title("RAG Chatbot Demo")

# Chat history - initialize only once
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload PDFs and I'll answer questions about them."}]
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.sidebar:
    st.title("PDF Upload")
    
    db_exists = test_rag.database_exists()
    
    st.write("Database Status:")
    st.write(f"ChromaDB exists: {db_exists}")
    
    if db_exists:
        st.success("The data is ready!")
    else:
        st.error("No data found")

    # Database management section
    st.subheader("Database Controls")

    # Clear database button
    if st.button("Clear Database"):
        try:
            test_rag.clear_chroma()
            st.success("Database cleared!")
            st.session_state.messages = [
                {"role": "assistant", "content": "Database has been cleared. Please upload new PDFs."}
            ]
            st.session_state.pdf_processed = False
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing database: {e}")

    # PDF upload section
    st.subheader("Upload PDF Files")
    pdf_docs = st.file_uploader(
        "Choose PDF files", 
        accept_multiple_files=True, 
        type="pdf",
        help="Upload one or more PDF files to create your knowledge base"
    )

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                success, num_chunks = test_rag.process_and_save_pdfs(pdf_docs)
                
                if success:
                    st.success(f"Successfully processed {len(pdf_docs)} PDFs with {num_chunks} total chunks!")
                    
                    # Update state
                    st.session_state.pdf_processed = True
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"Ready! I've processed {len(pdf_docs)} PDF files. Ask me anything about their content!"}
                    ]
                    st.rerun()
                else:
                    st.error("Failed to process PDFs. Check terminal for detailed errors.")
        else:
            st.warning("Please upload PDF files first.")

# Main chat interface
st.subheader("Chat with Your Documents")

# Display database info
if test_rag.database_exists():
    st.success("Database is ready! Ask questions about your uploaded PDFs.")
else:
    st.warning("No database found. Please upload PDFs using the sidebar.")

# Chat input
user_input = st.chat_input("Ask a question about your PDF documents...")

if user_input and user_input.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                if not test_rag.database_exists():
                    st.error("No database found. Please upload PDFs first.")
                    response_text = "No database available. Please upload PDF documents first."
                else:
                    formatted_response, response_text = test_rag.query_rag(user_input)
                    st.write(response_text)
                    
            except Exception as e:
                error_msg = f"Error querying database: {str(e)}"
                st.error(error_msg)
                response_text = error_msg
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Force rerun to show updated messages
    st.rerun()