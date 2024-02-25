"""
RAG 
"""

# Import necessary libraries
from typing import Type, Tuple, List
import streamlit as st
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from prompts.refactor_code_prompt import create_refactoring_prompt
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from openai import OpenAI
import os
import re
from io import BytesIO


def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """
    Parses the PDF file and extracts text.

    Args:
        file (BytesIO): PDF file object.
        filename (str): Name of the PDF file.

    Returns:
        Tuple[List[str], str]: List of extracted text and filename.
    """
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    """
    Converts extracted text into Document objects.

    Args:
        text (List[str]): List of extracted text.
        filename (str): Name of the PDF file.

    Returns:
        List[Document]: List of Document objects.
    """
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(doc)
    return doc_chunks


def docs_to_index(docs, api_key):
    """
    Converts Document objects into FAISS index.

    Args:
        docs: List of Document objects.
        openai_api_key: OpenAI API key.

    Returns:
        FAISS: FAISS index.
    """
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))
    # index = FAISS.from_documents(docs, embedding)
    return index


def get_index_for_pdf(pdf_files, pdf_names, api_key):
    """
    Gets FAISS index for PDF files.

    Args:
        pdf_files: List of PDF file objects.
        pdf_names: List of PDF file names.
        openai_api_key: OpenAI API key.

    Returns:
        FAISS: FAISS index.
    """
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, api_key)
    return index

# Cached function to create a vectordb for the provided PDF files
# @st.cache_data
def create_vectordb(files, filenames,api_key):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, api_key
        )
    return vectordb

def chat_with_openai (model,messages,api_key) : 
    client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=True
                    )
    return chat_completion

def chat_with_rag(api_key):
    """
    Refines the existing answer based on the provided context and question.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to refine the answer.

    Returns:
        The refined answer.
    """
   
    # Upload PDF files using Streamlit's file uploader
    pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

    # If PDF files are uploaded, create the vectordb and store it in the session state
    if pdf_files:
        pdf_file_names = [file.name for file in pdf_files]
        st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names,api_key)

    # Define the template for the chatbot prompt
    prompt_template = """
        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer short and to the point.
        
        The evidence are the context of the pdf extract with metadata. 
        
        Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
        
        Make sure to add filename and page number at the end of sentence you are citing to.
            
        Reply "Not applicable" if text is irrelevant.
        
        The PDF content is:
        {pdf_extract}
    """

    # Get the current prompt from the session state or set a default value
    prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

    # Display previous chat messages
    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Get the user's question using Streamlit's chat input
    question = st.chat_input("Ask anything")

    # Handle the user's question
    if question:
        vectordb = st.session_state.get("vectordb", None)
        if not vectordb:
            with st.chat_message("assistant"):
                st.write("You need to provide a PDF")
                st.stop()

        # Search the vectordb for similar content to the user's question
        search_results = vectordb.similarity_search(question, k=3)
        # st.write ("Search Results from VectorDB", search_results)
        # search_results
        pdf_extract = "/n ".join([result.page_content for result in search_results])

        # Update the prompt with the pdf extract
        prompt[0] = {
            "role": "system",
            "content": prompt_template.format(pdf_extract=pdf_extract),
        }

        # Add the user's question to the prompt and display it
        prompt.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Display an empty assistant message while waiting for the response
        with st.chat_message("assistant"):
            botmsg = st.empty()

        # Call ChatGPT with streaming and display the response as it comes
        response = []
        result = ""

    # st.write ( "Sending following Message", prompt)
        for chunk in chat_with_openai(
            model="gpt-3.5-turbo", messages=prompt, api_key=api_key
            ):
            text = chunk.choices[0].delta.content
            if text is not None:
                response.append(text)
                result = "".join(response).strip()
                botmsg.write(result)

        # Add the assistant's response to the prompt
        prompt.append({"role": "assistant", "content": result})

        # Store the updated prompt in the session state
        st.session_state["prompt"] = prompt
