import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv

load_dotenv()


## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"


## prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with tasks."),
    ("user", "{input}"),
])

def get_response(input, api_key, llm_model, temperature = 0.5, max_tokens = 1000):
    groq_api_key = api_key
    model = ChatGroq(
        model_name=llm_model,
        api_key=groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"input": input})


## streamlit app
st.title("Q&A Chatbot")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("API Key", type="password")
llm_model = st.sidebar.selectbox("Model", ["openai/gpt-oss-20b", "qwen/qwen3-32b", "openai/gpt-oss-120b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=10000, value=1000, step=100)

# Display conversation history at the top
if st.session_state.messages:
    st.subheader("Conversation History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input box at the bottom
if api_key:
    user_input = st.chat_input("Enter your question:")
    
    if user_input:
        # Add user question to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = get_response(user_input, api_key, llm_model, temperature, max_tokens)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the display
        st.rerun()
else:
    st.error("Please enter your API key to continue.")
