from dotenv import load_dotenv
import os 
load_dotenv()

import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq 

def load_pdf(pdf_path):                          
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300):  # smaller chunks = safer
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = model.encode(chunks)
    return embeddings

def build_vector_db(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search(query, index, chunks, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vec = model.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in indices[0]]

def build_safe_prompt(question, context_chunks, max_chars=3000):
    # keep adding chunks until we hit the char limit
    context = ""
    for chunk in context_chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n\n"
    
    prompt = f"""Answer the question using only the context below. Be concise.

Context:
{context}

Question: {question}
Answer:"""
    
    return prompt

def ask_llm(question, context_chunks):
    prompt = build_safe_prompt(question, context_chunks, max_chars=3000)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # try models one by one if one fails
    models = [
        "llama3-8b-8192",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
    ]
    
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            if "model" in str(e).lower() or "bad request" in str(e).lower():
                continue  # try next model
            raise e  # if different error, raise it
    
    return "Sorry, could not get an answer. Please try again."
