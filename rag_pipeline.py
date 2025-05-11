import os
import requests
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# Define the path for your FAISS vector DB
DB_FAISS_PATH = 'vector_db'

# Set up custom prompt template
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

# Retrieve the Hugging Face API key from Streamlit's secrets
HUGGING_FACE_API_KEY = st.secrets["huggingface"]["api_key"]

# Hugging Face Inference API URL for the Llama model
API_URL = "https://api-inference.huggingface.co/models/TheBloke/Llama-2-7B-Chat-GGML"

def load_llm():
    """Call the Hugging Face API to load the LLaMA model"""
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
    }
    
    # Making an API call to Hugging Face to confirm model availability
    response = requests.get(API_URL, headers=headers)
    
    if response.status_code == 200:
        print("Model loaded successfully!")
    else:
        print(f"Error: {response.status_code}")
        return None
    
    return API_URL  # Return the model URL to be used later in API requests

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
    if llm is None:
        return None
    prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, prompt, db)
    return qa

def final_result(query):
    """Query the Hugging Face model and return the result"""
    try:
        qa_chain = qa_bot()
        if qa_chain is None:
            return {"result": "Error: Model could not be loaded."}
        
        # Send a request to Hugging Face API with the user query
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"},
            json={"inputs": query}
        )

        # Parse the API response
        if response.status_code == 200:
            return {"result": response.json()}  # Assuming the API returns the answer in JSON format
        else:
            return {"result": f"Error: {response.status_code}"}
    except Exception as e:
        return {"result": f"Error: {str(e)}"}
