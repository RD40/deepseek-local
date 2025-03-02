import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Apply custom CSS for UI styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #ffffff; }
    h1 { color: #4ecdc4; text-align: center; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .stChatMessage { padding: 1.5rem; border-radius: 15px; margin: 1rem 0; max-width: 80%; }
    .stChatMessage[data-testid="user"] { background: rgba(78, 205, 196, 0.15); border: 1px solid #4ecdc4; margin-left: auto; }
    .stChatMessage[data-testid="assistant"] { background: rgba(30, 39, 73, 0.9); border: 1px solid #2a3a6e; margin-right: auto; }
    .stTextInput>div>div>input { background: rgba(255, 255, 255, 0.1) !important; color: white !important; border: 1px solid #4ecdc4 !important; border-radius: 10px; }
    .stButton>button { background: linear-gradient(45deg, #4ecdc4, #45b7af) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 10px 25px !important; transition: all 0.3s ease !important; }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 5px 15px rgba(78, 205, 196, 0.4); }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("Deepseek That Works")
st.caption("Since DeepSeek never works, it's better to use Ollama and run it locally.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model Selection with trimmed names
    model_options = ["deepseek-r1:1.5b", "deepseek-r1:7b"]
    selected_model = st.selectbox("Choose Model", model_options, index=0).strip()  # Ensuring no trailing spaces

    st.divider()
    st.markdown("### 🚀 Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert  
    - 🛠️ Debugging Assistant  
    - 📄 Code Documentation  
    - 💡 Solution Design  
    """)

    st.divider()
    st.markdown('Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)', unsafe_allow_html=True)

# Initialize chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I am DeepSeek. How can I help you code?"}]

# Chat message container
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Initialize Ollama Chat Model
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.7
)

# System Prompt for AI
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging coding-related problems. Always respond in English."
)

# Function to build chat history as a prompt chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Function to get AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    try:
        return processing_pipeline.invoke({})
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Sorry, something went wrong."

# Get user input
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    with st.spinner("Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
