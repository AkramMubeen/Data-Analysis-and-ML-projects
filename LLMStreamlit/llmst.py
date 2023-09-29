import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from templatehtml import css, bot_template, user_template

# Initialize text splitter for breaking down text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Function to extract text from PDF documents and split it into chunks
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    docs = text_splitter.split_text(text)
    return docs

# Function to create a vector store from text documents
def get_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain for chat
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.5, streaming=True)

    # Create a memory to store conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and chat interactions
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF CHAT",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Display the app header and input field for user questions
    st.header("PDF CHAT :books:")
    user_question = st.text_input("Query with your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar section for uploading and processing PDF documents
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                gif_runner = st.image("gif\giphy.gif")
                # Get text from PDF documents and split it into chunks
                text_chunks = get_pdf_text(pdf_docs)
                
                # Create a vector store from text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Create a conversation chain for chat
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                gif_runner.empty()
            st.success(body="Done! You may proceed querying the PDFs.", icon="🤖")

if __name__ == '__main__':
    main()