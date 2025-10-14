import streamlit as st
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Nexus Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'current_project' not in st.session_state:
    st.session_state.current_project = "Home"

# Project configurations
PROJECTS = {
    "Home": {
        "module": "home",
        "name": "Home Dashboard",
        "description": "Welcome to AI Nexus Hub",
        "icon": "🏠"
    },
    "EnhancedChatbot": {
        "module": "projects.chatbot",
        "name": "Enhanced Q&A Chatbot",
        "description": "Advanced chatbot with multiple LLM options and parameters",
        "icon": "🤖"
    },
    "ConversationalRAG": {
        "module": "projects.conversational_rag",
        "name": "Conversational RAG with PDF",
        "description": "Chat with uploaded PDFs using conversation history",
        "icon": "📚"
    },
    "SearchAgent": {
        "module": "projects.search_agent",
        "name": "Web Search Agent",
        "description": "Intelligent agent that searches web, arXiv, and Wikipedia",
        "icon": "🔍"
    },
    "URLSummarizer": {
        "module": "projects.url_summarizer",
        "name": "URL & YouTube Summarizer",
        "description": "Summarize content from websites and YouTube videos",
        "icon": "📺"
    }
}

def home_page():
    """Home page with project overview"""
    st.title("🚀 AI Nexus Hub")
    st.markdown("---")
    
    # API Key Status
    if st.session_state.groq_api_key:
        st.success("✅ Groq API Key: Configured")
    else:
        st.warning("⚠️ Please configure your Groq API Key in the sidebar")
    
    st.markdown("### 📋 Available Tools")
    st.markdown("---")
    
    # Display projects in a grid
    cols = st.columns(2)
    
    for i, (project_id, project_info) in enumerate(PROJECTS.items()):
        if project_id != "Home":
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"#### {project_info['icon']} {project_info['name']}")
                    st.write(project_info['description'])
        
                    
                    if st.button(f"Open {project_info['name']}", 
                                key=f"btn_{project_id}",
                                use_container_width=True):
                        st.session_state.current_project = project_id
                        st.rerun()
                    st.markdown("---")

def load_project(project_id):
    """Dynamically load and run the selected project"""
    try:
        if project_id == "Home":
            home_page()
            return
        
        project_config = PROJECTS.get(project_id)
        if not project_config:
            st.error(f"Project '{project_id}' not found!")
            return
        
        # Import and run the project
        module_path = project_config["module"]
        if module_path == "projects.chatbot":
            from projects.chatbot import run_chatbot
            run_chatbot()
        elif module_path == "projects.conversational_rag":
            from projects.conversational_rag import run_conversational_rag
            run_conversational_rag()
        elif module_path == "projects.search_agent":
            from projects.search_agent import run_search_agent
            run_search_agent()
        elif module_path == "projects.url_summarizer":
            from projects.url_summarizer import run_url_summarizer
            run_url_summarizer()
        
    except ImportError as e:
        st.error(f"Error loading project '{project_id}': {str(e)}")
        st.info("Make sure the project module is properly implemented")
    except Exception as e:
        st.error(f"Error running project '{project_id}': {str(e)}")

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 AI Nexus Hub")
        st.markdown("---")
        
        # API Key Configuration
        st.subheader("🔑 Groq API Configuration")
        api_key = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            value=st.session_state.groq_api_key,
            help="Get your API key from https://console.groq.com"
        )
        
        if api_key:
            st.session_state.groq_api_key = api_key
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your Groq API Key")
        
        st.markdown("---")
        
        # Project Selection
        st.subheader("🔧 Select Tool")
        
        # Create project selection buttons
        for project_id, project_info in PROJECTS.items():
            if project_id != "Home":
                if st.button(
                    f"{project_info['icon']} {project_info['name']}",
                    key=f"sidebar_{project_id}",
                    use_container_width=True,
                    type="primary" if st.session_state.current_project == project_id else "secondary"
                ):
                    st.session_state.current_project = project_id
                    st.rerun()
        
        st.markdown("---")
        
        # Current project info
        current_project_info = PROJECTS.get(st.session_state.current_project, {})
        st.subheader("ℹ️ Current Project")
        st.write(f"**{current_project_info.get('name', 'Home')}**")
        st.write(current_project_info.get('description', 'Select a project to get started'))
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.write(f"**Active Tool:** {current_project_info.get('name', 'Home')}")
        st.write(f"**API Key:** {'✅ Configured' if st.session_state.groq_api_key else '❌ Missing'}")

    # Main content area
    try:
        load_project(st.session_state.current_project)
    except Exception as e:
        st.error(f"Error loading project: {str(e)}")
        st.info("Please check the project implementation or try another project")

if __name__ == "__main__":
    main()