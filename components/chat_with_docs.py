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




def refine_chain(llm, pages, question):
    """
    Refines the existing answer based on the provided context and question.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to refine the answer.

    Returns:
        The refined answer.
    """
    # Define the refine prompt template
    refine_prompt_template = """
    The original question is: \n {question} \n
    The provided answer is: \n {existing_answer}\n
    Refine the existing answer if needed with the following context: \n {context_str} \n
    Given the extracted content and the question, create a final answer.
    If the answer is not contained in the context, say "answer not available in context. \n\n
    """
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )

    # Define the initial question prompt template
    initial_question_prompt_template = """
    Answer the question as precise as possible using the provided context only. \n\n
    Context: \n {context_str} \n
    Question: \n {question} \n
    Answer:
    """
    initial_question_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template=initial_question_prompt_template,
    )

    # Load the QA chain for refinement
    refine_chain = load_qa_chain(
        llm=llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=initial_question_prompt,
        refine_prompt=refine_prompt,
    )

    # Execute the refine chain
    refine_outputs = refine_chain({"input_documents": pages, "question": question})
    return refine_outputs['output_text']

def stuff_chain(llm, pages, question):
    """
    Retrieves an answer for the given question based on the provided context.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to answer.

    Returns:
        The answer to the question.
    """
    # Define the question prompt template
    question_prompt_template = """
    Use the following pieces of context to answer the question at the end. If you 
    don't know the answer, just say that you don't know, don't try to make up an 
    answer.
    Context: \n {context} \n
    Question: \n {query} \n
    Answer:
    """
    prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "query"]
    )

    # Load the QA chain for answering
    stuff_chain = load_qa_chain(
        llm=llm, chain_type="stuff", prompt=prompt, document_variable_name="context",
    )

    # Execute the QA chain for answering
    stuff_answer = stuff_chain(
        {"input_documents": pages, "query": question}, return_only_outputs=True
    )
    return stuff_answer["output_text"]

def map_reduce_chain(llm, pages, question):
    """
    Retrieves an answer for the given question based on the provided context using map-reduce approach.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to answer.

    Returns:
        The answer to the question.
    """
    # Define the question prompt template
    question_prompt_template = """
    Answer the question as precise as possible using the provided context. \n\n
    Context: \n {context} \n
    Question: \n {question} \n
    Answer:
    """
    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    # Define the combine prompt template
    combine_prompt_template = """Given the extracted content and the question, create a final answer.
    If the answer is not contained in the context, say "answer not available in context. \n\n
    Summaries: \n {summaries}?\n
    Question: \n {question} \n
    Answer:
    """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    # Load the QA chain for map-reduce approach
    map_reduce_chain = load_qa_chain(
        llm=llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        question_prompt=question_prompt,
        combine_prompt=combine_prompt,
    )

    # Execute the QA chain for map-reduce approach
    map_reduce_outputs = map_reduce_chain(
        {"input_documents": pages, "question": question}, return_only_outputs=True
    )
    return map_reduce_outputs["output_text"]

#Parse PDF file 
def parse_pdf(file: BytesIO) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output # type: ignore


def text_to_docs(text: List[str]) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=50,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for  chunk in chunks:
            doc = Document(
                page_content=chunk
            )
            doc_chunks.append(doc)
    return doc_chunks


# def get_documents_from_pdf(pdf_files):
#     text = []
#     for pdf_file in pdf_files:
#         text = parse_pdf(BytesIO(pdf_file.getvalue()))
#         # print(text)
#         text = text + (text_to_docs(text))
#     return text

def get_documents_from_pdf(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        text = parse_pdf(BytesIO(pdf_file.getvalue()))
        # if isinstance(text, str):
        #     text = [text]
        documents = documents + text_to_docs(text) # type: ignore
    return documents


def chat_with_docs(chat: Type[ChatOpenAI]):
    """
    Display the RefactorRite page with a title, description, and code input form.

    Parameters:
    - openai_api_key (str): API key for accessing the OpenAI GPT-3.5 model.
#     """
#     st.set_page_config(
#     page_title='GenAI use cases',
#     layout="wide",
#     # initial_sidebar_state="expanded",
# )

    chat_hists =  st.session_state.get("chat_history", [{"user" : "system", "assistant" : ""}])
  
    
    st.title ( " Question Answering with Large Documents using LangChain ")
    st.subheader(" This notebook demonstrates how to build a question-answering (Q&A) system using LangChain (load_qa_chain) ")
    # Upload PDF files
    pdf_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    if pdf_files:
        documents = []
        documents = get_documents_from_pdf(pdf_files)

        # Print the documents (this will be displayed in the terminal if running locally)
        # print(documents[0].page_content)

        col1, col2 = st.columns(2)
        with col1:
            container = st.container(border=True)
            container.subheader('Your Text ')
            # col1.text_area("",documents,height=400)
            container.write(documents)
        with col2:
            with st.form("myform") : 
                st.subheader('Your Chat')
                st.subheader("Ask your Question here ?")
                prompt = st.text_input("How to make veg curry")
                option = st.selectbox ( 
                            label = "Choose retrieval pattern",
                            options= ('Stuff Chain','Map reduce','refine'),
                            index=None, 
                            )
                submitted = st.form_submit_button("Chat with AI")

        if prompt and option and submitted :

            # st.session_state["chat_history"] = prompt
            # llm = ChatOpenAI(temperature=0)
            # st.write("you select option is " , option)
            answer=stuff_chain(chat, documents, prompt)
            # st.session_state["chat_history"] = answer
            curr_chat = {
                "user" : prompt,
                "assistant" : answer
                }
            # print(type(chat_hists))
            chat_hists.append(curr_chat)
            st.session_state.chat_history = chat_hists
            
            if chat_hists != None : 
                for chat in chat_hists: 
                    if chat["user"] != "system" : 
                        for key in chat:
                            with col2.chat_message(key) :
                                st.write(chat[key]) 