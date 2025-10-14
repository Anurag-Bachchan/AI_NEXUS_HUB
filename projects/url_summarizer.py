import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_community.document_loaders import WebBaseLoader

def run_url_summarizer():
    """URL & YouTube Summarizer - Summarize content from websites and YouTube videos"""
    st.title("📺 URL & YouTube Summarizer")
    st.markdown("Summarize content from any website URL or YouTube video")
    
    # Check API key
    if not st.session_state.get('groq_api_key'):
        st.error("⚠️ Please configure your Groq API key in the sidebar first!")
        return
    
    # URL input
    url = st.text_input("Enter YouTube or Website URL", placeholder="https://...")
    
    # Summarization prompt
    prompt_template = """
    Provide a comprehensive summary of the following content in about 300 words:
    Content: {text}

    SUMMARY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    if st.button("Summarize Content", type="primary"):
        if not url.strip():
            st.error("Please enter a URL")
        elif not validators.url(url):
            st.error("Please enter a valid URL")
        else:
            try:
                with st.spinner("Loading and summarizing content..."):
                    llm = ChatGroq(api_key=st.session_state.groq_api_key, model="llama-3.3-70b-versatile")
                    
                    # Load content based on URL type
                    data = None
                    if "youtube.com" in url or "youtu.be" in url:
                        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                        data = loader.load()
                    else:
                        loader = WebBaseLoader(url)
                        data = loader.load()
                    
                    if data and len(data) > 0:
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        summary = chain.run(data)
                        
                        st.success("Summary Generated!")
                        st.write(summary)
                        
                        # Show metadata
                        if hasattr(data[0], 'metadata'):
                            with st.expander("View Source Info"):
                                st.json(data[0].metadata)
                    else:
                        st.error("No content could be loaded from the URL")
                        
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("""
                Troubleshooting tips:
                • For YouTube: Ensure the video has captions
                • For websites: Some sites block automated access
                • Try a different URL
                """)