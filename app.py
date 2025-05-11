import streamlit as st
from rag_pipeline import final_result
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import torch
torch.classes.__path__ = []  # Neutralizes the path inspection


# Page configuration
st.set_page_config(page_title="Apple Product Chatbot", layout="wide")

# Title at the top-left
st.markdown("<h1 style='text-align: left;'>Apple Product Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask questions about Apple products based on the extracted knowledge base.")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for history
st.sidebar.title("Chat History")
if st.session_state.history:
    for i, question in enumerate(reversed(st.session_state.history), 1):
        st.sidebar.markdown(f"{i}. {question}")
else:
    st.sidebar.markdown("No previous queries.")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt
if query := st.chat_input("Type your question here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append(query)

    # Display user input
    st.chat_message("user").markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = final_result(query)
                answer = response["result"]
            except Exception as e:
                answer = f"Error: {str(e)}"
        st.markdown(answer)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
