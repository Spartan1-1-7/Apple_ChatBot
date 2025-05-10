import torch
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

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
    """Load the quantized LLaMA model via CTransformers"""
    llm = CTransformers(
        model='llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        max_new_tokens=512,
        temperature=0.5,
        device='cuda'  # Use GPU (if available)
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    """Create a RetrievalQA chain with custom prompt"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 1}),  # Reduce chunk to prevent token overflow
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    """Load the FAISS vector DB and prepare the QA chain"""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'}  # Move to GPU (if available)
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
