import streamlit as st
from streamlit_option_menu import option_menu
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from components import (
    home,
    refactor_page,
    style_page,
    test_page,
    lang_page,
    code_documentation_page,
    database_page,
    refactor_page,
    chat_with_docs,
    chat_with_rag
)
import os
from langchain.embeddings.openai import OpenAIEmbeddings

def llm_chat():
    llm = ChatOpenAI(temperature=0)
    return llm


def llm_embedding():
    llm = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return llm



def main():
    # load_dotenv()
    print(os.getenv("OPENAI_API_KEY"))
    api_key=os.getenv("OPENAI_API_KEY")

    chat = llm_chat()
    # embedding = llm_embedding()
    st.set_page_config(
        page_title="CodeCraft GPT: A Comprehensive Code Enhancement Platform",
        page_icon="🚀",
        layout="wide"
    )
    
    with st.sidebar:
        selected = option_menu(
            menu_title="GEN AI Patterns",
            options=[
                "Home", "Chat with Documents" , "Chat Documents with RAG" , "Search Documents", "Text Summarization",
                "TestGenius", "LangLink", "CodeDocGenius", "Database"
            ],
            icons=[
                'house', 'gear', 'gear', 'palette', 'clipboard2-pulse', 'clipboard2-pulse',
                'code-slash', 'file-text', 'database'
            ],
            default_index=0
        )
    
    # Dictionary containing functions without invoking them
    pages = {
        "Chat with Documents" : chat_with_docs.chat_with_docs,
        "Chat Documents with RAG" : chat_with_rag.chat_with_rag,
        "Search Documents": refactor_page.show_refactor_page,
        "Text Summarization": style_page.show_style_page,
        "TestGenius": test_page.show_test_page,
        "LangLink": lang_page.show_lang_page,
        "CodeDocGenius": code_documentation_page.show_doc_page,
        "Database": database_page.show_database_page,
        "Home": home.show_home_page  # Removed the () for immediate call
    }

    if selected in pages:
        if selected == "Chat Documents with RAG" : 
            pages[selected](api_key)
        elif selected != "Home":
            # Call the function corresponding to the selected page
            pages[selected](chat)
        else:
            home.show_home_page()
            st.info("⚠️ Please select an option from the side menu to proceed.")
    else:
        st.error("Page not found!")

if __name__ == "__main__":
    main()
