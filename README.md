# Apple ChatBot

## Overview
The Apple ChatBot is a project designed to provide an interactive chatbot experience for exploring Apple product specifications and details. It leverages advanced AI models and a retrieval-augmented generation (RAG) pipeline to deliver accurate and context-aware responses.

## Features
- **AI-Powered Chatbot**: Utilizes the LLaMA 2 model for natural language understanding and generation.
- **Product Data Integration**: Includes detailed technical specifications for a wide range of Apple products, such as iPhones, MacBooks, and more.
- **Retrieval-Augmented Generation (RAG)**: Combines a vector database with a generative model to provide precise and relevant answers.

## Project Structure
- `app.py`: Main application file for running the chatbot.
- `rag_pipeline.py`: Implements the RAG pipeline for integrating the vector database with the generative model.
- `ingest_db.ipynb`: Jupyter Notebook for ingesting and preprocessing data into the vector database.
- `vector_db/`: Contains the FAISS index and related files for the vector database.
- `apple_products_data/`: Directory with Apple product data in PDF format.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd apple_chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download LLaMA 2 model in the working directory.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dependencies
- Python 3.12
- FAISS
- PyTorch
- LLaMA 2 model

## Usage
- Start the chatbot by running `app.py`.
- Interact with the chatbot to query Apple product specifications.

## Data Sources
The project includes technical specifications for various Apple products, stored in the `apple_products_data/` directory.

