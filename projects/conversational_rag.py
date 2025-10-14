from dotenv import load_dotenv
import os
import tempfile
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

def run_conversational_rag():
    st.title("📚 Conversational RAG with PDF")
    st.markdown("Upload PDF files and chat interactively with memory & Groq LLM")

    # Retrieve keys
    groq_api_key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    # Check environment setup
    if not groq_api_key:
        st.error("⚠️ Missing Groq API Key. Set GROQ_API_KEY in .env or sidebar.")
        return
    if not hf_token:
        st.error("⚠️ Missing Hugging Face token (HF_TOKEN) in .env.")
        return

    # Initialize HuggingFace embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"use_auth_token": hf_token}
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {e}")
        return

    # Initialize Groq LLM (no tenant argument)
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )
    except Exception as e:
        st.error(f"❌ Failed to connect to Groq: {e}")
        return

    # Initialize session state
    if "store" not in st.session_state:
        st.session_state.store = {}

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ RAG Settings")
        session_id = st.text_input("Session ID", value="default_session")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if session_id in st.session_state.store:
                del st.session_state.store[session_id]
            st.success("Chat history cleared!")

    # Upload PDF section
    uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files and "vectorstore" not in st.session_state:
        with st.spinner("🔍 Processing PDFs... Please wait."):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                os.unlink(tmp_path)

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # Build vectorstore
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Processed {len(uploaded_files)} PDFs into {len(splits)} chunks.")

    # Chat section
    if "vectorstore" in st.session_state:
        retriever = st.session_state.vectorstore.as_retriever()

        # Contextual chain setup
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history and the latest question, rephrase it as a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Display chat history
        session_history = get_session_history(session_id)
        for msg in session_history.messages:
            st.chat_message(msg.type).write(msg.content)

        # Chat input
        user_input = st.chat_input("💬 Ask a question about your PDFs...")
        if user_input:
            response = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write(response["answer"])
    else:
        st.info("📄 Please upload PDF files to start chatting.")

if __name__ == "__main__":
    run_conversational_rag()
