import openai_secret_manager
import requests
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# Access your Hugging Face API key from Streamlit secrets
HUGGING_FACE_API_KEY = st.secrets["huggingface"]["api_key"]

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
    """Load the model via Hugging Face API"""
    model_id = "TheBloke/Llama-2-7B-Chat-GGML"  # Model repo ID on Hugging Face
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
    }
    payload = {
        "inputs": "Hello, world!"  # A sample input for testing the model
    }
    
    # Make an API call to Hugging Face model for inference
    response = requests.post(f"https://api-inference.huggingface.co/models/{model_id}", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()  # Assuming a valid response for testing
    else:
        raise Exception(f"Error loading model: {response.status_code} - {response.text}")

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
    llm = load_llm()  # This will now load the model via the API call
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
