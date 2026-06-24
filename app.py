import streamlit as st
from rag_engine import load_pdf, chunk_text, build_vector_db, search, ask_llm
from sentence_transformers import SentenceTransformer
import tempfile
import numpy as np

st.set_page_config(
    page_title="PDF Chat Assistant",
    
    layout="wide"
)

st.title(" PDF Chat Assistant")
st.markdown("*Upload any PDF and ask questions about it using AI*")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(" Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Max size: 3GB per file"
    )

    # only process if new files uploaded
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        
        # only re-process if files changed
        if "loaded_files" not in st.session_state or st.session_state.loaded_files != file_names:
            
            with st.spinner(" Reading PDFs..."):
                text = ""
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(uploaded_file.read())
                        text += load_pdf(f.name)

            chunks = chunk_text(text, chunk_size=300)
            st.info(f" Found {len(chunks)} chunks, now embedding...")
            progress = st.progress(0)

            model = SentenceTransformer('all-MiniLM-L6-v2')
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_embeddings = model.encode(batch)
                all_embeddings.append(batch_embeddings)
                progress.progress(min(i + batch_size, len(chunks)) / len(chunks))

            embeddings = np.vstack(all_embeddings)
            index = build_vector_db(embeddings)
            progress.empty()

            # save everything to session state
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.text = text
            st.session_state.loaded_files = file_names
            st.session_state.ready = True  #  flag that PDF is ready

        st.success(f" {len(uploaded_files)} PDF(s) loaded!")
        for f in uploaded_files:
            st.write(f" {f.name}")
        st.divider()
        st.metric("Total Chunks", len(st.session_state.chunks))
        st.metric("Total Words", len(st.session_state.text.split()))
        st.metric("Total PDFs", len(uploaded_files))

with col2:
    st.subheader(" Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    #  only show chat input AFTER PDF is fully ready
    if "ready" not in st.session_state or not st.session_state.ready:
        st.info(" Please upload a PDF first to start chatting!")
    else:
        question = st.chat_input("Ask anything about your PDFs...")

        if question:
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    relevant_chunks = search(question, st.session_state.index, st.session_state.chunks)
                    answer = ask_llm(question, relevant_chunks)
                st.write(answer)

                with st.expander(" Source chunks used"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1}:** {chunk[:200]}...")

            st.session_state.messages.append({"role": "assistant", "content": answer})
