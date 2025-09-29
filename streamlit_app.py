import streamlit as st
import os
import pandas as pd 

# --- LangChain/RAG Imports ---
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Using HuggingFace's Inference API for the LLM and Embeddings (more stable)
from langchain_community.llms import HuggingFaceHub 
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. RAG SYSTEM SETUP (Loads the database and LLM once) ---

@st.cache_resource
def setup_rag():
    # 1. Initialize the HuggingFace Embeddings (free inference API)
    # Note: You MUST set the HUGGINGFACEHUB_API_TOKEN as a secret!

    # We need a stable embedding model:
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
    )

    # 2. Load the persisted OHS vector store from the folder
    vectorstore = Chroma(
        persist_directory="./ohs_chroma_db", 
        embedding_function=embeddings
    )

    # 3. Initialize the LLM (using the free tier HuggingFace model)
    # The flan-t5-large model is small and fast, perfect for free tier RAG
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.1, "max_length": 512},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
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
