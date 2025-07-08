import streamlit as st
import pandas as pd
import PyPDF2
import docx
import os
import re
import zipfile
import tempfile
import shutil
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        st.error(f"Could not read PDF {pdf_path}: {e}") # Changed to st.error
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Could not read DOCX {docx_path}: {e}") # Changed to st.error
    return text

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    text = ""
    try:
        # Try reading with default comma delimiter, skipping bad lines
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        text = df.to_string()
    except pd.errors.ParserError as e:
        st.error(f"Could not read CSV {csv_path} due to parsing error: {e}. Attempting to read as plain text.") # Changed to st.error
        try:
            # If pandas fails, try reading as plain text line by line
            with open(csv_path, "r", encoding="utf-8", errors='ignore') as file:
                text = file.read()
        except Exception as e_plain:
            st.error(f"Could not read CSV {csv_path} as plain text: {e_plain}") # Changed to st.error
    except Exception as e:
        st.error(f"Could not read CSV {csv_path}: {e}") # Changed to st.error
    return text

# Function to extract text from Excel
def extract_text_from_excel(excel_path):
    text = ""
    try:
        df = pd.read_excel(excel_path)
        text = df.to_string()
    except Exception as e:
        st.error(f"Could not read Excel {excel_path}: {e}") # Changed to st.error
    return text

# Function to extract text from TXT
def extract_text_from_txt(txt_path):
    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        st.error(f"Could not read TXT {txt_path}: {e}") # Changed to st.error
    return text

# Function to process all files in the corpus directory
def process_corpus(corpus_dir):
    knowledge_base = []
    # st.info(f"Scanning directory: {corpus_dir} and its subdirectories...") # Removed verbose message

    for root, _, files in os.walk(corpus_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_text = ""
            # st.info(f"Processing file: {filepath}") # Removed verbose message
            if filename.endswith(".pdf"):
                file_text = extract_text_from_pdf(filepath)
                # st.info(f"  - Detected as PDF.") # Removed verbose message
            elif filename.endswith(".docx"):
                file_text = extract_text_from_docx(filepath)
                # st.info(f"  - Detected as DOCX.") # Removed verbose message
            elif filename.endswith(".csv"):
                file_text = extract_text_from_csv(filepath)
                # st.info(f"  - Detected as CSV.") # Removed verbose message
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_text = extract_text_from_excel(filepath)
                # st.info(f"  - Detected as Excel.") # Removed verbose message
            elif filename.endswith(".txt"):
                file_text = extract_text_from_txt(filepath)
                # st.info(f"  - Detected as TXT.") # Removed verbose message
            else:
                # st.info(f"  - Skipping unsupported file type: {filename}") # Removed verbose message
                continue

            if not file_text:
                # st.warning(f"  - Extracted text is empty for {filename}.") # Removed verbose message
                pass # Keep silent unless there's an actual error
            else:
                # Split text into lines for line number tracking
                lines = file_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():  # Only add non-empty lines
                        try:
                            # Generate embedding for the content chunk
                            embedding_response = genai.embed_content(model="models/embedding-001", content=line.strip())
                            embedding = embedding_response['embedding']
                            knowledge_base.append({
                                "filename": os.path.relpath(filepath, corpus_dir), # Store relative path
                                "line_number": i + 1,
                                "content": line.strip(),
                                "embedding": embedding
                            })
                        except Exception as e:
                            st.error(f"Error generating embedding for {filename} line {i+1}: {e}")


    return knowledge_base

# Function to answer queries using LLM with RAG
def answer_query(query, knowledge_base, llm_model, embedding_model, top_k=5):
    if not knowledge_base:
        return "I could not find any relevant information in the provided documents to answer your question."

    try:
        # Generate embedding for the query
        query_embedding_response = genai.embed_content(model="models/embedding-001", content=query)
        query_embedding = query_embedding_response['embedding']

        # Calculate cosine similarity with all document embeddings
        similarities = []
        for i, item in enumerate(knowledge_base):
            if "embedding" in item:
                similarity = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(item["embedding"]).reshape(1, -1))[0][0]
                similarities.append((similarity, i))

        # Sort by similarity and get top_k results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k_indices = [idx for sim, idx in similarities[:top_k]]

        # Construct context for the LLM from top_k results
        context = ""
        retrieved_items = []
        for idx in top_k_indices:
            item = knowledge_base[idx]
            retrieved_items.append(item)
            context += f"File: {item['filename']}, Line: {item['line_number']}\nContent: {item['content']}\n\n"

        if not context:
            return "I could not find any relevant information in the provided documents to answer your question."

        # Generation step: Use LLM to answer based on context
        prompt = f"""You are a helpful assistant that answers questions based on the provided documents. 
        If the answer is not available in the context, state that you don't have enough information.

        Context from documents:
        {context}

        Question: {query}
        Answer:"""
        
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating a response from the AI model: {e}"

# Streamlit UI
st.set_page_config(page_title="Document Chatbot", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 Document Chatbot")
st.markdown("""
    Welcome to the Document Chatbot! Upload your document corpus as a ZIP file.
    This chatbot uses Google Gemini to answer your questions based on the content of your documents.
""")

# Gemini API Key input
gemini_api_key = st.text_input("Enter your Google Gemini API Key (from Google AI Studio):", type="password", key="gemini_api_key_input")

# Initialize Gemini model
model = None
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        embedding_model = genai.GenerativeModel('models/embedding-001') # Initialize embedding model
        st.success("Gemini models initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize Gemini model. Please check your API key: {e}")
else:
    st.warning("Please enter your Gemini API Key to enable AI-powered responses.")

# File uploader for ZIP file
uploaded_file = st.file_uploader("Upload your document corpus (ZIP file)", type="zip")

if uploaded_file is not None:
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, uploaded_file.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            st.success(f"Successfully extracted {uploaded_file.name} to a temporary directory.")

            # Process the extracted files
            with st.spinner("Processing documents from ZIP... This may take a moment."):
                st.session_state.knowledge_base = process_corpus(temp_dir)
            st.success("Documents processed!")
            if not st.session_state.knowledge_base:
                st.warning("No supported documents found in the uploaded ZIP file. Please ensure your ZIP contains PDF, DOCX, CSV, XLSX, or TXT files.")
            # Clear chat history when a new directory is processed
            st.session_state.messages = []

        except zipfile.BadZipFile:
            st.error("Invalid ZIP file. Please upload a valid .zip archive.")
        except Exception as e:
            st.error(f"An error occurred during ZIP extraction or processing: {e}")

# Initialize knowledge_base if not already present
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []

# Display initial message if no documents have been processed yet
if not st.session_state.knowledge_base and uploaded_file is None:
    st.info("Please upload a ZIP file containing your document corpus to begin.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            if not model:
                response = "Please enter a valid Gemini API Key to enable AI-powered responses."
            elif not st.session_state.knowledge_base:
                response = "I cannot answer questions as no documents were processed. Please upload a ZIP file with your documents."
            else:
                response = answer_query(prompt, st.session_state.knowledge_base, model, embedding_model)
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1.  **Get your Gemini API Key:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to create your API key.
2.  **Enter your API Key:** Paste your Gemini API key into the input box on the main page.
3.  **Prepare your documents:** Place all your PDF, DOCX, CSV, XLSX, and TXT files into a single folder.
4.  **Compress the folder:** Create a `.zip` archive of this folder.
5.  **Upload the ZIP file:** Use the "Upload your document corpus (ZIP file)" button to upload your `.zip` file.
6.  **Ask questions** in the chat input box. The chatbot will use the uploaded documents and Gemini to provide intelligent answers.
""")