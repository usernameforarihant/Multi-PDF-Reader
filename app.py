# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from htmlTemplates import css,bot_template,user_template
# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader=PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+=page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter=CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks=text_splitter.split_text(text)
#     return chunks

# def get_vector_store(chunks):
#     embeddings=OpenAIEmbeddings()
#     vectorstore=FAISS.from_texts(texts=chunks,embedding=embeddings)
#     return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def handle_userinput(user_question):
#     response=st.session_state.conversation({'question':user_question})
#     # st.write(response)
#     st.session_state.chat_history=response['chat_history']
#     for i,message in enumerate(st.session_state.chat_history):
#         if i%2==0:
#             st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiplt PDFs",page_icon=":books:")
#     st.write(css,unsafe_allow_html=True)
#     if 'conversation' not in st.session_state:
#         st.session_state.coversation=None
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history=None
#     st.header('Chat with multiple PDFs "books:')
#     user_question=st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     st.write(user_template.replace("{{MSG}}","hello Pix"),unsafe_allow_html=True)
#     st.write(bot_template.replace("{{MSG}}","hello human"),unsafe_allow_html=True)
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs=st.file_uploader("Upload your Pdfs and click on 'Process' ",accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 #get pdf text
#                 raw_text=get_pdf_text(pdf_docs)
#                 # st.write(raw_text)


#                 #get pdf text chunks
#                 text_chunks=get_text_chunks(raw_text)
#                 # st.write(text_chunks)

#                 #vector store
#                 vector_store=get_vector_store(text_chunks)

#                 #create conversation chain (takes history of conversation and gives next element in convo)
#                 # conversation=get_conversation(vector_store)
#                 st.session_state.conversation=get_conversation_chain(vector_store)
    
# if __name__=='__main__':
#     main()
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    # Page Layout
    st.title(':books: Chat with Multiple PDFs')
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

    # Sidebar for Document Upload
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click 'Process'", 
            accept_multiple_files=True, 
            type=['pdf']
        )
        process_button = st.button("Process", use_container_width=True)

        if process_button:
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    # Extract PDF Text
                    raw_text = get_pdf_text(pdf_docs)

                    # Split into Chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create Vector Store
                    vector_store = get_vector_store(text_chunks)

                    # Conversation Chain
                    st.session_state.conversation = get_conversation_chain(vector_store)

                st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    # User Interaction Area
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question about the uploaded documents:")

    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.error("Please upload and process the PDFs first.")

    # Footer Message
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-size: 14px;">
            Built with ❤️ using Streamlit and LangChain
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
