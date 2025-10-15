# 🧠 **AI Nexus Hub — Multi-Tool GenAI App**

An advanced **GenAI Streamlit application** powered by **Groq API**, integrating multiple intelligent tools for conversational AI, document analysis, information retrieval, and content summarization — all in one unified interface.

---

## 🚀 **Features**

### 🗣️ **1. Advanced Chatbot**
A fully customizable chatbot that lets you:
- Choose between multiple **LLMs** (via Groq API)
- Adjust **parameters** like temperature, max tokens, etc.
- Maintain chat history across turns

🧩 *File:* `projects/chatbot.py`

---

### 📚 **2. Conversational RAG with PDF**
Chat directly with uploaded PDFs using **Retrieval-Augmented Generation (RAG)**.  
- Upload one or more PDFs  
- Ask contextual questions  
- Enjoy persistent **conversation memory**

🧩 *File:* `projects/conversational_rag.py`

---

### 🌐 **3. Web Search Agent**
An intelligent multi-source search agent that:
- Retrieves information from the **web**, **Wikipedia**, and **arXiv**
- Summarizes results into concise, AI-generated insights
- Acts as an AI research assistant

🧩 *File:* `projects/search_agent.py`

---

### 📰 **4. URL & YouTube Summarizer**
Summarize content from:
- **Webpages** (via URL)
- **YouTube videos** (via transcript parsing or metadata)

🧩 *File:* `projects/url_summarizer.py`

---

## 🧩 **Project Structure**

<img width="256" height="313" alt="Screenshot 2025-10-14 014214" src="https://github.com/user-attachments/assets/6bcbfd02-825b-45eb-bbe1-ad8b09a93546" />


---

## 🔑 **Environment Variables**

Create a `.env` file in the root directory with your **Groq API Key**:

> ⚠️ The app requires a valid **Groq API key** to use any of the tools.

---

## 🧰 **Installation**

### **1. Clone the repository**
```bash
git clone https://github.com/Anurag-Bachchan/AI_NEXUS_HUB
cd AI_NEXUS_HUB
```

### **2. Install Dependencies**

> pip install -r requirements.txt
---

## 🏆 **Tech Stack**
- Frontend / UI: Streamlit

- Backend: Python

- AI Frameworks: LangChain, Groq API

- Vector Search: FAISS / LangChain embeddings

- Deployment: Render

----

## 💡 **Example Usage**

- Select a tool from the sidebar

- Enter your Groq API key

- Start chatting, searching, or summarizing instantly!

