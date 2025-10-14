import streamlit as st
from langchain_groq import ChatGroq 
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler 

def run_search_agent():
    """Web Search Agent - Intelligent agent that searches web, arXiv, and Wikipedia"""
    st.title("🔍 Web Search Agent")
    st.markdown("AI agent that can search the web, arXiv papers, and Wikipedia to answer your questions")
    
    # Check API key
    if not st.session_state.get('groq_api_key'):
        st.error("⚠️ Please configure your Groq API key in the sidebar first!")
        return
    
    # Initialize tools
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250))
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250))
    search = DuckDuckGoSearchRun(name="Search")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can search the web, arXiv, and Wikipedia. What would you like to know?"}
        ]
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # User input
    if prompt := st.chat_input("Ask me to search for something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        try:
            # Initialize agent
            llm = ChatGroq(
                groq_api_key=st.session_state.groq_api_key, 
                model_name="llama-3.3-70b-versatile", 
                streaming=True
            )
            tools = [search, arxiv, wikipedia]
            search_agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
            )
            
            # Run agent
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write(response)
                
        except Exception as e:
            st.error(f"Error: {e}")