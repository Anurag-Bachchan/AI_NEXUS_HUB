import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

def run_chatbot():
    """Enhanced Q&A Chatbot with chat history context"""
    st.title("🤖 Enhanced Q&A Chatbot")
    st.markdown("Chat with an intelligent AI assistant that remembers your conversation history")
    
    # Check API key
    if not st.session_state.get('groq_api_key'):
        st.error("⚠️ Please configure your Groq API key in the sidebar first!")
        return
    
    # Create columns for settings in main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Chat Interface")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'human':
                with st.chat_message("user"):
                    st.write(f"**You:** {message['content']}")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**Assistant:** {message['content']}")
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
        
        # User input
        user_input = st.chat_input("Ask me anything...")
        
        if user_input:
            try:
                # Initialize LLM with current settings
                llm = ChatGroq(
                    groq_api_key=st.session_state.groq_api_key,
                    model_name=st.session_state.get('chatbot_model', 'llama-3.3-70b-versatile'),
                    temperature=st.session_state.get('chatbot_temperature', 0.7),
                    max_tokens=st.session_state.get('chatbot_max_tokens', 150)
                )
                
                # Build conversation history for context
                messages = []
                
                # System message
                messages.append(("system", "You are a helpful assistant. Provide accurate and concise responses. Use the conversation history to provide contextual answers."))
                
                # Add recent chat history (last 6 exchanges to avoid token limits)
                recent_history = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
                
                for message in recent_history:
                    if message['type'] == 'human':
                        messages.append(("human", message['content']))
                    else:
                        messages.append(("ai", message['content']))
                
                # Add current question
                messages.append(("human", user_input))
                
                # Create prompt with history
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | llm | StrOutputParser()
                
                # Generate response
                with st.spinner("Thinking..."):
                    response = chain.invoke({})
                
                # Update chat history
                st.session_state.chat_history.append({
                    'type': 'human', 
                    'content': user_input
                })
                st.session_state.chat_history.append({
                    'type': 'ai', 
                    'content': response
                })
                
                # Display new messages immediately
                with st.chat_message("user"):
                    st.write(f"**You:** {user_input}")
                with st.chat_message("assistant"):
                    st.write(f"**Assistant:** {response}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        # Settings in main content area (not sidebar)
        st.subheader("⚙️ Chat Settings")
        
        # Model selection
        llm_model = st.selectbox(
            "Select LLM Model",
            ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"],
            index=0,
            key="chatbot_model"
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            help="Lower = more deterministic, Higher = more creative",
            key="chatbot_temperature"
        )
        
        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens", 
            min_value=50, 
            max_value=1000, 
            value=150,
            help="Maximum length of the response",
            key="chatbot_max_tokens"
        )
        
        # History context slider
        history_context = st.slider(
            "History Context (messages)",
            min_value=2,
            max_value=12,
            value=6,
            help="Number of previous messages to use as context",
            key="history_context"
        )
        
        # Chat controls
        st.markdown("---")
        st.subheader("🛠️ Controls")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset Settings", use_container_width=True):
                if 'chatbot_model' in st.session_state:
                    del st.session_state.chat_model
                if 'chatbot_temperature' in st.session_state:
                    del st.session_state.chatbot_temperature
                if 'chatbot_max_tokens' in st.session_state:
                    del st.session_state.chatbot_max_tokens
                if 'history_context' in st.session_state:
                    del st.session_state.history_context
                st.success("Settings reset to defaults!")
                st.rerun()
        
        # Export chat history
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.chat_history:
                chat_text = "Chat History:\n\n"
                for i, message in enumerate(st.session_state.chat_history):
                    prefix = "You: " if message['type'] == 'human' else "Assistant: "
                    chat_text += f"{prefix}{message['content']}\n\n"
                
                st.download_button(
                    label="Download Chat History",
                    data=chat_text,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No chat history to export")
        
        # Current settings display
        st.markdown("---")
        st.subheader("📊 Current Settings")
        st.write(f"**Model:** {llm_model}")
        st.write(f"**Temperature:** {temperature}")
        st.write(f"**Max Tokens:** {max_tokens}")
        st.write(f"**History Context:** {history_context} messages")
        st.write(f"**Total Messages:** {len(st.session_state.chat_history)}")
    
    # Instructions section
    st.markdown("---")
    with st.expander("ℹ️ How to use this chatbot"):
        st.markdown("""
        **Enhanced Features:**
        - 💬 **Contextual Conversations**: The AI remembers your chat history and uses it to provide better answers
        - 📝 **Follow-up Questions**: Ask questions like "What did I say earlier?" or "Can you summarize our conversation?"
        - ⚙️ **Customizable Context**: Adjust how much history the AI considers
        - 💾 **Export Chats**: Download your entire conversation
        
        **Settings Explanation:**
        - **Temperature**: Controls creativity (0 = deterministic, 1 = creative)
        - **Max Tokens**: Limits response length
        - **History Context**: Number of previous messages used for context
        - **LLM Model**: Different models have different capabilities
        
        **Examples of contextual understanding:**
        - "What was my previous question about?"
        - "Based on what we discussed, what would you recommend?"
        - "Can you summarize our conversation so far?"
        - "Referring to my earlier point about..."
        
        **Tips:**
        - Use lower temperature for factual questions
        - Use higher temperature for creative tasks
        - Increase history context for longer, more coherent conversations
        - Adjust max tokens based on desired response length
        """)

# Alternative version with more advanced history management
def run_advanced_chatbot():
    """Advanced version with smarter history management"""
    st.title("🤖 Advanced Contextual Chatbot")
    
    # Initialize advanced chat history structure
    if "advanced_chat_history" not in st.session_state:
        st.session_state.advanced_chat_history = []
    
    # Enhanced system prompt that explicitly uses history
    system_prompt = """You are a helpful assistant that maintains context across conversations. 
    Use the chat history provided to understand the context of the current question. 
    If the user refers to previous messages, use the history to provide relevant answers.
    Be natural and maintain conversation flow."""
    
    user_input = st.chat_input("Ask me anything with context...")
    
    if user_input:
        # Build messages with history
        messages = [("system", system_prompt)]
        
        # Add conversation history
        for msg in st.session_state.advanced_chat_history:
            role = "human" if msg['type'] == 'human' else "ai"
            messages.append((role, msg['content']))
        
        # Add current message
        messages.append(("human", user_input))
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name=st.session_state.get('chatbot_model', 'llama-3.3-70b-versatile'),
            temperature=st.session_state.get('chatbot_temperature', 0.7)
        )
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        
        with st.spinner("Thinking with context..."):
            response = chain.invoke({})
        
        # Update history
        st.session_state.advanced_chat_history.extend([
            {'type': 'human', 'content': user_input},
            {'type': 'ai', 'content': response}
        ])
        
        # Display
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(response)

# For testing the project individually
if __name__ == "__main__":
    # Set a dummy API key for testing
    st.session_state.groq_api_key = "test-key"
    run_chatbot()