"""
RefactorRite - Code Refactoring Advisor

Leverage AI-driven code analysis and automated refactoring to enhance code
readability, boost performance, and improve maintainability. RefactorRite
suggests intelligent refinements and even automates the refactoring process,
allowing developers to focus on building robust software.
"""
from typing import Type
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from prompts.refactor_code_prompt import create_refactoring_prompt
from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from io import BytesIO
from typing import Tuple, List
from pypdf import PdfReader
import re
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter



def image_to_text(chat: Type[ChatOpenAI]):
    """
   Image to text LLM

    Parameters:
    - openai_api_key (str): API key for accessing the OpenAI GPT-3.5 model.
#     """
#     st.set_page_config(
#     page_title='GenAI use cases',
#     layout="wide",
#     # initial_sidebar_state="expanded",
# )


  
    
    st.title ( " Image to Text LLM Demo ")
    st.subheader("Upload your Image , LLM will describe the image ")
    # Upload PDF files
    image_file = st.file_uploader("Upload Image Files", type=".jpeg", accept_multiple_files=True)

    