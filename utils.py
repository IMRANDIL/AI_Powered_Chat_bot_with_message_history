import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

## setup streamlit

st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload pdf's and chat with the content!")

## Input the Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")

## Check if groq api key is provided

if api_key:
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
    
    ## Chat interface below
    session_id = st.text_input("Session ID", value="default_session")
    
    ## statefully manage chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)
    
    ## process uploaded PDF's
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temppdf)  # Clean up temporary file
            
        ## Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents=documents)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vector_store.as_retriever()
        
        ## contextualize prompts system
        contextulize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        ## contextualize_q_prompt
        contextulize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextulize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        ## history aware retriever
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextulize_q_prompt)
        
        ## Answer question 
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        ## get session history function
        def get_session_history(session_id: str) -> ChatMessageHistory:
            history = ChatMessageHistory()
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    history.add_user_message(message.content)
                elif isinstance(message, AIMessage):
                    history.add_ai_message(message.content)
            return history
        
        ## conversational rag chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message.type):
                st.write(message.content)
        
        # Chat input
        user_input = st.chat_input("Your question:")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            # Display user message
            with st.chat_message("human"):
                st.write(user_input)
            
            # Get AI response
            with st.chat_message("ai"):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )
                ai_message = response['answer']
                st.write(ai_message)
            
            # Add AI response to chat history
            st.session_state.chat_history.append(AIMessage(content=ai_message))
            
            
else:
    st.warning("Please enter your API key!")