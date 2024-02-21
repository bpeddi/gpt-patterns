import streamlit as st
from streamlit_option_menu import option_menu
from langchain.chat_models import ChatOpenAI
from components import (
    home,
    refactor_page,
    style_page,
    test_page,
    lang_page,
    code_documentation_page,
    database_page
)


def main():
    chat = None
    st.set_page_config(
        page_title="CodeCraft GPT: A Comprehensive Code Enhancement Platform",
        page_icon="🚀",
        layout="wide"
    )
    
    with st.sidebar:
        selected = option_menu(
            menu_title="GEN AI Patterns",
            options=[
                "Home", "Search Documents", "Text Summarization",
                "TestGenius", "LangLink", "CodeDocGenius", "Database"
            ],
            icons=[
                'house', 'gear', 'palette', 'clipboard2-pulse',
                'code-slash', 'file-text', 'database'
            ],
            default_index=0
        )
    
    # Dictionary containing functions without invoking them
    pages = {
        "Search Documents": refactor_page.show_refactor_page,
        "Text Summarization": style_page.show_style_page,
        "TestGenius": test_page.show_test_page,
        "LangLink": lang_page.show_lang_page,
        "CodeDocGenius": code_documentation_page.show_doc_page,
        "Database": database_page.show_database_page,
        "Home": home.show_home_page  # Removed the () for immediate call
    }

    if selected in pages:
        if selected != "Home":
            # Call the function corresponding to the selected page
            pages[selected](chat)
        else:
            home.show_home_page()
            st.info("⚠️ Please select an option from the side menu to proceed.")
    else:
        st.error("Page not found!")

if __name__ == "__main__":
    main()
