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

def chunk_text(text, chunk_size=1000):
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

def ask_llm(question, context_chunks):
    # use only 1 chunk, trimmed to 300 chars
    context = context_chunks[0][:300]
    
    prompt = f"""Answer briefly using only this context:

Context: {context}

Question: {question}
Answer:"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",   # ✅ bigger context window than llama3
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content