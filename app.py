import streamlit as st
import pandas as pd
import PyPDF2
import docx
import os
import re

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
        st.warning(f"Could not read PDF {pdf_path}: {e}")
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.warning(f"Could not read DOCX {docx_path}: {e}")
    return text

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    text = ""
    try:
        df = pd.read_csv(csv_path)
        text = df.to_string()
    except Exception as e:
        st.warning(f"Could not read CSV {csv_path}: {e}")
    return text

# Function to extract text from Excel
def extract_text_from_excel(excel_path):
    text = ""
    try:
        df = pd.read_excel(excel_path)
        text = df.to_string()
    except Exception as e:
        st.warning(f"Could not read Excel {excel_path}: {e}")
    return text

# Function to extract text from TXT
def extract_text_from_txt(txt_path):
    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        st.warning(f"Could not read TXT {txt_path}: {e}")
    return text

# Function to process all files in the corpus directory
def process_corpus(corpus_dir="corpus"):
    knowledge_base = []
    for filename in os.listdir(corpus_dir):
        filepath = os.path.join(corpus_dir, filename)
        if os.path.isfile(filepath):
            file_text = ""
            if filename.endswith(".pdf"):
                file_text = extract_text_from_pdf(filepath)
            elif filename.endswith(".docx"):
                file_text = extract_text_from_docx(filepath)
            elif filename.endswith(".csv"):
                file_text = extract_text_from_csv(filepath)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_text = extract_text_from_excel(filepath)
            elif filename.endswith(".txt"):
                file_text = extract_text_from_txt(filepath)
            else:
                st.info(f"Skipping unsupported file type: {filename}")
                continue

            if file_text:
                # Split text into lines for line number tracking
                lines = file_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():  # Only add non-empty lines
                        knowledge_base.append({
                            "filename": filename,
                            "line_number": i + 1,
                            "content": line.strip()
                        })
    return knowledge_base

# Function to answer queries
def answer_query(query, knowledge_base):
    results = []
    query_lower = query.lower()

    for item in knowledge_base:
        if query_lower in item["content"].lower():
            results.append(item)

    if not results:
        return "I could not find any relevant information in the provided documents."
    else:
        response = "Here's what I found:\n\n"
        for item in results:
            response += f"**File:** {item['filename']}\n"
            response += f"**Line Number:** {item['line_number']}\n"
            response += f"**Content:** {item['content']}\n\n"
        return response

# Streamlit UI
st.set_page_config(page_title="Document Chatbot", layout="wide")

st.title("📄 Document Chatbot")
st.markdown("""
    Welcome to the Document Chatbot! Upload your documents (PDF, DOCX, CSV, Excel, TXT)
    to the `corpus` folder and ask questions about their content.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process corpus only once
if "knowledge_base" not in st.session_state:
    with st.spinner("Processing documents... This may take a moment."):
        st.session_state.knowledge_base = process_corpus("C:/Users/polic/OneDrive/Desktop/chatbot_project/corpus")
    st.success("Documents processed!")
    if not st.session_state.knowledge_base:
        st.warning("No supported documents found in the 'corpus' folder. Please add some files to get started.")

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
        with st.spinner("Searching documents..."):
            if st.session_state.knowledge_base:
                response = answer_query(prompt, st.session_state.knowledge_base)
            else:
                response = "I cannot answer questions as no documents were processed. Please add files to the 'corpus' folder."
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1.  **Place your files** (PDF, DOCX, CSV, XLSX, TXT) into the `corpus` folder located at:
    `C:/Users/polic/OneDrive/Desktop/chatbot_project/corpus`
2.  **Rerun the application** (if already running) or start it. The chatbot will automatically process the files.
3.  **Ask questions** in the chat input box. The chatbot will search through your documents and provide relevant information, including the filename and line number for text-based files.
""")