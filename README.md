#  PDF Chat Assistant

Chat with any PDF using AI — powered by RAG architecture.

🔗 **Live Demo:** [app-link.streamlit.app](https://rag-project-2ikiqbmemls2bazp5yw7pu.streamlit.app/)

##  What is this?
Upload any PDF and ask questions about it in natural language.
The app finds the most relevant sections and answers using AI.

##  How it works
1. Upload a PDF
2. App splits it into chunks
3. Converts chunks to embeddings using SentenceTransformers
4. Stores in FAISS vector database
5. Your question is matched to relevant chunks using cosine similarity
6. LLaMA answers using only your document as context

##  Tech Stack
| Tool | Purpose |
|---|---|
| Streamlit | UI |
| SentenceTransformers | Text embeddings |
| FAISS | Vector similarity search |
| Groq + LLaMA3 | LLM inference |
| PyMuPDF | PDF text extraction |

##  Run Locally
git clone https://github.com/speco29/rag-project
cd rag-project
pip install -r requirements.txt
streamlit run app.py

##  Project Structure
rag-project/
├── app.py           # Streamlit UI
├── rag_engine.py    # RAG pipeline
├── requirements.txt
└── .streamlit/
    └── config.toml  # Upload size config
