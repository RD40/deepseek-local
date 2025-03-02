import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory

import os
import tempfile

api_key=os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.title("RAMESH.AI")
st.title("Conversational RAG with PDF Uploads & Chat History")
st.write("Upload PDFs and interact with their content.")

# Input Groq API Key

if api_key:
    # Initialize embedding model and vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b")
    DOCUMENT_VECTOR_DB = InMemoryVectorStore(embeddings)

    # Session management
    if "store" not in st.session_state:
        st.session_state.store = {}

    session_id = st.text_input("Session ID", value="default_session")

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # PDF Upload Section
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error loading PDF {uploaded_file.name}: {e}")
            finally:
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        DOCUMENT_VECTOR_DB.add_documents(splits)

        st.success("✅ PDFs processed successfully! You can now ask questions.")

    retriever = DOCUMENT_VECTOR_DB.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the question for standalone understanding, considering chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the retrieved context to answer concisely. If unsure, say 'I don't know'.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
    )

    user_input = st.text_input("Ask a question about the uploaded PDFs:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input}, config={"configurable": {"session_id": session_id}}
        )
        st.write("Assistant:", response['answer'])
else:
    st.warning("Please enter your Groq API Key")
