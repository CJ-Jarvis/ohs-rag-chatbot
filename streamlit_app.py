import streamlit as st
import os
import pandas as pd # <-- Required to work with your OHS data (though pandas isn't explicitly used here, it's good practice)

# --- LangChain/RAG Imports ---
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. RAG SYSTEM SETUP (Loads the database and LLM once) ---

@st.cache_resource
def setup_rag():
    # 1. Initialize the free embedding model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_url="https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
)
    
    # 2. Load the persisted OHS vector store from the folder
    # This path is relative to the streamlit_app.py file, which is correct.
    vectorstore = Chroma(
        persist_directory="./ohs_chroma_db", 
        embedding_function=embeddings
    )
    
    # 3. Initialize the LLM (using the free tier Gemini model)
    # It pulls the key securely from the environment variable set via secrets.toml
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # 4. Define the Retrieval Chain Components
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    
    # Define the Prompt
    system_prompt = (
        "You are an expert South African OHS Compliance Assistant. "
        "Answer the user's question only based on the provided OHS context. "
        "If the context does not contain the answer, clearly state that the answer is not available in the OHS knowledge base. "
        "Always reference the 'Source/Category' from the metadata in your final answer."
        "\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- 2. STREAMLIT INTERFACE (Starts running here) ---

# Check for API Key (Prevents crash by stopping execution gracefully if key is missing)
if "GEMINI_API_KEY" not in os.environ and "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ **API Key Missing:** Please ensure the `GEMINI_API_KEY` is set in your Streamlit secrets.")
    st.stop()

# Initialize the RAG chain
retrieval_chain = setup_rag()

st.title("🇿🇦 OHS Compliance Assistant")
st.caption("Answers based on OHS Act, Regulations, and 2024/2025 Amendments.")

# Chat history management
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input and RAG execution
if prompt := st.chat_input("Ask a question about South African OHS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching OHS regulations..."):
        try:
            response = retrieval_chain.invoke({"input": prompt})
            actual_answer = response['answer']
        except Exception as e:
            # Displays a generic error if the RAG system fails during query execution
            actual_answer = f"An internal error occurred during retrieval. Please check the logs."
            
    with st.chat_message("assistant"):
        st.markdown(actual_answer)
    st.session_state.messages.append({"role": "assistant", "content": actual_answer})

