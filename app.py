import streamlit as st
from rag_engine import load_pdf, chunk_text, build_vector_db, search, ask_llm
from sentence_transformers import SentenceTransformer
import tempfile
import numpy as np

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄", 
    layout="wide"
)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## PDF Chat Assistant")
    st.markdown("*Chat with your documents using AI*")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Max size: 3GB per file"
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]

        if "loaded_files" not in st.session_state or st.session_state.loaded_files != file_names:

            with st.spinner("Reading PDFs..."):
                text = ""
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(uploaded_file.read())
                        text += load_pdf(f.name)

            chunks = chunk_text(text, chunk_size=300)

            model = SentenceTransformer('all-MiniLM-L6-v2')
            batch_size = 32
            all_embeddings = []
            total_batches = len(range(0, len(chunks), batch_size))

            with st.status("Processing PDF...", expanded=True) as status:
                st.write("Reading text...")

                for idx, i in enumerate(range(0, len(chunks), batch_size)):
                    batch = chunks[i:i+batch_size]
                    all_embeddings.append(model.encode(batch))

                    percent = int((idx + 1) / total_batches * 100)
                    st.write(f"Embedding chunks... {percent}%")

                status.update(label="PDF Ready!", state="complete", expanded=False)

            embeddings = np.vstack(all_embeddings)
            index = build_vector_db(embeddings)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.text = text
            st.session_state.loaded_files = file_names
            st.session_state.ready = True

        st.success(f"{len(uploaded_files)} PDF(s) ready!")
        st.markdown("---")
        st.metric("PDFs", len(uploaded_files))
        st.metric("Chunks", len(st.session_state.chunks))
        st.metric("Words", f"{len(st.session_state.text.split()):,}")

        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Upload New PDF"):
            for key in ["chunks", "index", "text", "loaded_files", "ready", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("---")
        st.markdown("### Uploaded Files")
        for f in uploaded_files:
            size_mb = f.size / (1024*1024)
            st.markdown(f"**{f.name}** ({size_mb:.1f} MB)")

# ---- MAIN AREA ----
if "ready" not in st.session_state or not st.session_state.ready:
    st.markdown("""
    <div style='text-align: center; padding: 80px;'>
        <h1>Welcome to PDF Chat</h1>
        <h3>Upload a PDF and let the magic begin</h3>
        <br><br>
        <p style='font-size: 18px;'>Supports multiple PDFs at once</p>
        <p style='font-size: 18px;'>Up to 3GB per file</p>
        <p style='font-size: 18px;'>Powered by LLaMA3 AI</p>
        
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("### Chat with your PDFs")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # chat input
    question = st.chat_input("Ask anything about your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")

            relevant_chunks = search(question, st.session_state.index, st.session_state.chunks)
            answer = ask_llm(question, relevant_chunks)

            placeholder.empty()
            st.write(answer)

            with st.expander("Source chunks used"):
                for i, chunk in enumerate(relevant_chunks):
                    st.write(f"**Chunk {i+1}:** {chunk[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
