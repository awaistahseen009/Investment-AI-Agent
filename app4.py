# import streamlit as st
# import os
# import dotenv
# import uuid
# import logging
# from pathlib import Path

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # SQLite configuration for Linux environments
# if os.name == 'posix':
#     __import__('pysqlite3')
#     import sys
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage, AIMessage
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from test4 import (
#     load_doc_to_db,
#     stream_llm_response,
#     stream_llm_rag_response,
#     initialize_vector_db,
# )

# dotenv.load_dotenv()

# # Page configuration
# st.set_page_config(
#     page_title="RAG-Enhanced Chat",
#     page_icon="📚",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # Header
# st.header("RAG-Enhanced Chat Application")
# st.subheader("Upload documents or URLs to enhance the conversation with relevant context")

# # Initialize session state
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())
#     logger.info(f"New session started: {st.session_state.session_id}")

# if "rag_sources" not in st.session_state:
#     st.session_state.rag_sources = []

# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Hello! I'm ready to help. You can upload documents or provide URLs to enhance our conversation with relevant context."}
#     ]

# # Initialize vector database if not already loaded
# chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
# if os.path.exists(chroma_db_path) and any(Path(chroma_db_path).iterdir()):
#     vector_db = Chroma(
#         persist_directory=chroma_db_path,
#         embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     )
#     print("Already set")
#     st.session_state.vector_db = vector_db
# else:
#     st.session_state.vector_db = None
#     st.warning("Please upload documents or provide URLs to enable context-aware responses.")

# # Sidebar configuration
# with st.sidebar:
#     # API Key Management
#     if "GOOGLE_API_KEY" not in os.environ:
#         with st.popover("🔐 API Key Configuration"):
#             google_api_key = st.text_input(
#                 "Enter your Google API Key",
#                 value=os.getenv("GOOGLE_API_KEY", ""),
#                 type="password",
#                 key="google_api_key",
#             )
#             if not google_api_key:
#                 st.warning("Please enter your Google API Key to continue")
#     else:
#         google_api_key = os.environ["GOOGLE_API_KEY"]

#     st.divider()

#     if st.button("Clear Chat", type="primary"):
#         st.session_state.messages = [st.session_state.messages[0]]
#         st.rerun()

#     # Document Upload Section
#     st.header("Knowledge Base")

#     # File upload
#     uploaded_files = st.file_uploader(
#         "📄 Upload Documents",
#         type=["pdf", "txt", "docx", "md"],
#         accept_multiple_files=True,
#         help="Supported formats: PDF, TXT, DOCX, MD",
#         key="rag_docs",
#         on_change=load_doc_to_db
#     )

#     # Show loaded sources with additional information

#     with st.expander(f"📚 Knowledge Base Sources ({len(st.session_state.rag_sources)})"):
#         if st.session_state.vector_db is not None:
#             if st.session_state.get("collections"):
#                 for doc_name, info in st.session_state.collections.items():
#                     col1, col2 = st.columns([3, 1])
#                     with col1:
#                         st.write(f"📄 {doc_name}")
#                     with col2:
#                         if st.button("🗑️", key=f"delete_{doc_name}", help=f"Delete {doc_name}"):
#                             try:
#                                 st.session_state.vector_db._client.delete_collection(name=info["collection_name"])
#                                 del st.session_state.collections[doc_name]
#                                 st.session_state.rag_sources.remove(doc_name)
#                                 st.experimental_rerun()
#                             except Exception as e:
#                                 st.error(f"Error deleting collection: {e}")
#             else:
#                 st.write("No sources loaded yet")

# # Main chat interface
# if not google_api_key and "GOOGLE_API_KEY" not in os.environ:
#     st.warning("⚠️ Please configure your API key in the sidebar to continue")
# else:
#     try:
#         llm_stream = ChatGoogleGenerativeAI(
#             model="gemini-1.5-pro",
#             temperature=0.4,
#             streaming=True,
#             google_api_key=google_api_key,
#         )

#         # Display chat messages
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         # Chat input
#         if prompt := st.chat_input("Type your message here..."):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             # Generate response
#             with st.chat_message("assistant"):
#                 messages = [
#                     HumanMessage(content=m["content"]) if m["role"] == "user"
#                     else AIMessage(content=m["content"])
#                     for m in st.session_state.messages
#                 ]

#                 if st.session_state.vector_db is None:
#                     st.write_stream(stream_llm_response(llm_stream, messages))
#                 else:
#                     st.write_stream(stream_llm_rag_response(llm_stream, messages))

#     except Exception as e:
#         logger.error(f"Error in main chat interface: {e}")
#         st.error(f"An error occurred: {str(e)}")

# # Footer
# with st.sidebar:
#     st.divider()
#     st.markdown("""
#     ### About
#     This application combines the power of LLMs with RAG (Retrieval Augmented Generation) 
#     to provide context-aware responses based on your documents and web content.
#     """)
#app.py
import streamlit as st
import os
from dotenv import load_dotenv
import uuid
import logging
from pathlib import Path
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# For SQLite configuration in Linux
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from test4 import load_doc_to_db, stream_llm_response, stream_llm_rag_response, load_url_to_db

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "12345"

def initialize_session_state():
    """Initialize all session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []
        st.session_state.documents_metadata = {}
    if "url_metadata" not in st.session_state:
        st.session_state.url_metadata = {}
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    
    if "vector_db" not in st.session_state:
        chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_db_path) and any(Path(chroma_db_path).iterdir()):
            print("Initializing existing database")
            st.session_state.vector_db = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            )
        else:
            st.session_state.vector_db = None
    
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_login" not in st.session_state:
        st.session_state.show_login = False
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

def check_admin_auth():
    """Authenticate admin users."""
    if st.session_state.get("show_login", False):
        if not st.session_state.authenticated:
            col1, col2 = st.columns([1, 1])
            username = col1.text_input("Username")
            password = col2.text_input("Password", type="password")

            if st.button("Login"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.is_admin = True
                    st.session_state.show_login = False
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            return False
    return True

def logout_user():
    """Log out the user by clearing authentication-related session state."""
    st.session_state.authenticated = False
    st.session_state.is_admin = False
    st.session_state.show_login = False
    # st.session_state.rag_sources = []
    # st.session_state.vector_db = None
    # st.rerun()

def handle_url_metadata(url):
    """Handle URL content metadata collection and submission."""
    st.subheader("URL Content Metadata")

    # Retrieve metadata from session state
    # metadata = st.session_state.get("url_metadata", {})
    # url = metadata.get("url", "").strip()
    if not url:
        st.warning("No URL provided in metadata. Please enter a valid URL.")
        return False

    # Form to collect metadata
    with st.form(key="url_metadata_form"):
        # Input fields for metadata
        url_title = st.text_input("Title", key="url_title_key")
        url_industry = st.selectbox(
            "Industry",
            [
                "Technology", "Healthcare", "Finance", "Manufacturing",
                "Retail", "Energy", "Entertainment", "Real Estate", "Other"
            ],
            
            key="url_industry_key"
        )
        url_category = st.selectbox(
            "Category",
            ["Technical", "Business", "Legal", "Other"],
            key="url_category_key"
        )
        url_description = st.text_area("Description", key="url_description_key")

        # Submit button for validating metadata fields
        metadata_valid = st.form_submit_button("Validate Metadata")

    # Check if metadata is validated
    if metadata_valid:
        if not url_title or not url_industry or not url_category or not url_description:
            st.error("Please complete all fields in the form before submitting.")
            return False

        # Save the updated metadata in session state
        st.session_state.url_metadata = {
            "title": url_title,
            "category": url_category,
            "industry": url_industry,
            "description": url_description,
            "url": url,
            "uploaded_by": "admin",  # Or dynamically set the username
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_type": "url"
        }

        st.success("Metadata validated successfully. Click 'Add URL Content' to submit.")

    # Button outside the form to submit the metadata
    if st.button("Add URL Content", key="add_url_button_key"):
        # Ensure metadata is already validated and exists in session state
        metadata = st.session_state.get("url_metadata")
        if not metadata:
            st.error("Metadata validation is incomplete. Please validate metadata first.")
            return False

        # Call the function to load the URL metadata into the database
        try:
            load_url_to_db()  # Assumes this function is defined elsewhere
            st.success(f"URL content added successfully: {url}")
        except Exception as e:
            st.error(f"Failed to add URL content: {e}")
            return False

        return True

    return False



def handle_document_upload(uploaded_files):
    """Handle document uploads and metadata storage."""
    with st.form(key="doc_metadata_form"):
        doc_industry = st.selectbox(
            "Industry",
            ["Technology", "Healthcare", "Finance", "Manufacturing", 
             "Retail", "Energy", "Entertainment", "Real Estate", "Other"],
            key="doc_industry_select"
        )
        doc_category = st.selectbox(
            "Category",
            ["Technical", "Business", "Legal", "Other"],
            key="doc_category_select"
        )
        doc_description = st.text_area("Description", key="doc_description_input")
        
        submit_button = st.form_submit_button("Upload Documents")
        
        if submit_button:
            for doc in uploaded_files:
                metadata = {
                    "filename": doc.name,
                    "industry": doc_industry,
                    "category": doc_category,
                    "description": doc_description,
                    "uploaded_by": ADMIN_USERNAME,
                    "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_type": doc.type,
                    "source_type": "document"
                }
                st.session_state.documents_metadata[doc.name] = metadata
            load_doc_to_db()
            st.success("Documents uploaded successfully")

def display_knowledge_base():
    """Display the knowledge base with enhanced metadata."""
    with st.expander("📚 Knowledge Base"):
        for source in st.session_state.rag_sources:
            metadata = st.session_state.documents_metadata.get(source, {})
            source_title = metadata.get('title', source)
            if metadata.get('filename'):
                source_title = metadata.get('filename')
                
            st.markdown(f"### {source_title}")
            st.write(f"Source Type: {metadata.get('source_type', 'Unknown')}")
            st.write(f"Industry: {metadata.get('industry', 'Not specified')}")
            st.write(f"Category: {metadata.get('category', 'Not specified')}")
            st.write(f"Description: {metadata.get('description', 'No description provided')}")
            st.write(f"Uploaded: {metadata.get('upload_date', 'Unknown')}")
            
            if metadata.get('url'):
                st.write(f"URL: {metadata.get('url')}")

            if st.button("Delete", key=f"del_{source}"):
                st.session_state.rag_sources.remove(source)
                del st.session_state.documents_metadata[source]
                st.rerun()

def handle_chat():
    """Handle chat interaction."""
    llm_stream = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.2,
        streaming=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            if st.session_state.vector_db is None:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Investment Assistant", page_icon="📚", layout="centered")
    initialize_session_state()

    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Ultimate Investiment Assistant")
    with col2:
        if st.button("Admin"):
            st.session_state.show_login = True
            st.rerun()

    if st.session_state.show_login and not check_admin_auth():
        return

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.button("Logout", on_click=logout_user)

            if st.session_state.is_admin:
                # Document upload section
                st.subheader("📄 Upload Documents")
                uploaded_files = st.file_uploader(
                    "Select Files",
                    type=["pdf", "txt", "docx", "md"],
                    accept_multiple_files=True,
                    help="Supported formats: PDF, TXT, DOCX, MD",
                    key="rag_docs",
                )
                if uploaded_files:
                    handle_document_upload(uploaded_files)

                # URL upload section
                st.subheader("🌐 Add Web Content")
                url_input = st.text_input(
                    "Enter URL",
                    placeholder="https://example.com",
                    help="Enter a URL to include web content",
                    key="rag_url"
                )
                if url_input:
                    handle_url_metadata(url_input)

                display_knowledge_base()

    if st.session_state.vector_db is not None:
        handle_chat()
    else:
        st.warning("No Chroma database found. Please upload documents as an admin to initialize the database.")

if __name__ == "__main__":
    main()