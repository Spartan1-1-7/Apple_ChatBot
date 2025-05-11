import os
import streamlit as st
import requests
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vector_db'

custom_prompt_template = """
Use the following pieces of information to answer the user's question. 
If you don't know the answer, just say you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    """Create a custom prompt template for QA"""
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

def load_llm():
    """Load the model via Hugging Face API using your token"""
    # Get the Hugging Face API token from Streamlit secrets
    hf_api_token = st.secrets["huggingface"]["api_token"]  # Fetch token from secrets

    if not hf_api_token:
        raise ValueError("Hugging Face API Token is missing. Please set it in Streamlit secrets.")

    # Set up the Hugging Face API headers
    headers = {
        "Authorization": f"Bearer {hf_api_token}"  # Use the token for authentication
    }

    model_url = "https://api-inference.huggingface.co/models/TheBloke/Llama-2-7B-Chat-GGML"
    
    def query_model(input_text):
        """Function to query the model via Hugging Face API"""
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_length": 512,         
                "temperature": 0.7,        
                "top_p": 0.9,              
                "top_k": 50,               
            }
        }
        response = requests.post(model_url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}

    return query_model

def retrieval_qa_chain(llm, prompt, db):
    """Create a RetrievalQA chain with custom prompt"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 1}),  
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    """Load the FAISS vector DB and prepare the QA chain"""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, prompt, db)
    return qa

def final_result(query):
    """Query the QA chain and return the result"""
    try:
        qa_chain = qa_bot()
        response = qa_chain.invoke({'query': query})
        return response
    except Exception as e:
        return {"result": f"Error: {str(e)}"}
