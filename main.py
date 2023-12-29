import os
import streamlit as st
from streamlit_option_menu import option_menu
from components import (
    home,
    refactor_page,
    style_page,
    test_page,
    lang_page,
    code_documentation_page,
    database_page
)

def set_openai_api_key():
    st.sidebar.title("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API key:")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.warning("Please enter your OpenAI API key.")

def main():
    st.set_page_config(
        page_title="CodeCraft GPT: A Comprehensive Code Enhancement Platform",
        page_icon="🚀",
        layout="wide"
    )

    # Check if OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        set_openai_api_key()

    # Display sidebar items only if API key is set
    if "OPENAI_API_KEY" in os.environ:
        with st.sidebar:
            selected = option_menu(
                menu_title="CodeCraftGPT",
                options=[
                    "Home", "RefactorRite", "StyleSculpt", "TestGenius", 
                    "LangLink", "CodeDocGenius", "Database"
                ],
                icons=[
                    'house', 'gear', 'palette', 'clipboard2-pulse', 
                    'code-slash', 'file-text', 'database'
                ],
                default_index=0
            )

        # Display pages based on selection
        pages = {
            "RefactorRite": refactor_page.show_refactor_page,
            "StyleSculpt": style_page.show_style_page,
            "TestGenius": test_page.show_test_page,
            "LangLink": lang_page.show_lang_page,
            "CodeDocGenius": code_documentation_page.show_doc_page,
            "Database": database_page.show_database_page,
            "Home": home.show_home_page
        }

        if selected in pages:
            pages[selected]()
        else:
            st.error("Page not found!")
    else:
        st.warning("Please enter your OpenAI API key to access pages.")

if __name__ == "__main__":
    main()
