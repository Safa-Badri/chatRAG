# -------------------------------
# Install packages if not yet:
# pip install transformers torch psycopg[binary] google-genai numpy PyPDF2 python-docx
# -------------------------------

import os
import psycopg
from psycopg.rows import dict_row
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import time
from google.api_core import exceptions as api_exceptions
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx

# -------------------------------
# Configurations
data_folder = "C:\\Chatbot-RAG\\data\\TRANS_TXT"
db_connection_str = "dbname='rag_chatbot' user='postgres' password='safa' host='localhost' port='5433'"
embedding_model_name = "BAAI/bge-large-en"
genai.configure(api_key="")  # Ton API Key

# -------------------------------
# Initialize embedding model
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# -------------------------------
# Helper functions to read files
def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="cp1252") as f:
        return f.read()

# -------------------------------
# Functions for embeddings
def calculate_embeddings(text: str) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding[0].cpu().numpy().tolist()

def save_embedding(corpus: str, embedding: list[float], cursor) -> None:
    cursor.execute(
        "INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)",
        (corpus, embedding)
    )

# -------------------------------
# Vector search function
def search_similar_docs(question: str, cursor, top_k: int = 3) -> list[dict]:
    question_embedding = calculate_embeddings(question)
    
    # Convert Python list to PostgreSQL vector literal
    vector_str = "[" + ",".join([str(x) for x in question_embedding]) + "]"
    
    cursor.execute(
        f"SELECT corpus, embedding <#> '{vector_str}'::vector AS distance "
        f"FROM embeddings ORDER BY embedding <#> '{vector_str}'::vector LIMIT {top_k};"
    )
    
    results = cursor.fetchall()
    return results


# -------------------------------
# Generate answer with Gemini
def generate_answer(question, context=""):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    max_retries = 5
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        
        except api_exceptions.ResourceExhausted:
            print(f"Quota exceeded (429). Retrying in {base_delay}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(base_delay)
            base_delay *= 2
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            
    return "Error: Could not generate an answer due to API quota limits after multiple retries."

# -------------------------------
# Main execution: create table & insert embeddings
with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("DROP TABLE IF EXISTS embeddings;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                corpus TEXT,
                embedding VECTOR(1024)
            );
        """)

        # Process files
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                file_path = os.path.join(root, file)
                text = read_txt(file_path)
                if not text.strip():
                    continue
                # Optional: split long texts into chunks
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                for chunk in chunks:
                    embedding = calculate_embeddings(chunk)
                    save_embedding(chunk, embedding, cur)
                    print(f"Saved chunk of {file} (length {len(chunk)})")

# -------------------------------
# Example RAG usage
with psycopg.connect(db_connection_str) as conn:
    with conn.cursor(row_factory=dict_row) as cur:
        user_question = "Comment je peux s'inscrire à English Connection?"
        # Retrieve similar docs
        top_docs = search_similar_docs(user_question, cur, top_k=3)
        context = "\n".join([doc["corpus"] for doc in top_docs])
        # Generate answer
        answer = generate_answer(user_question, context)
        print("\n=== RAG GEMINI ANSWER ===")
        print(answer)
