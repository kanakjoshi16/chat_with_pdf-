import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from htmlTemplates import css, reply_template
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# first of all we need to extract text from pdfs
def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# then we convert text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        task_type="retrieval_query"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question from the provided context, if the answer is not in
    provided context just say, "answer is not available in the context".\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=1)
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def handle_query(user_question):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    fiass_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = fiass_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    reply = response["output_text"]
    
    st.write(css,unsafe_allow_html=True)
    st.write(reply_template.replace("{{reply}}", reply),unsafe_allow_html=True)

def main():
    load_dotenv()
    key=os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)

    st.set_page_config(page_title="Chat with PDF",
                       page_icon=":books:")
    
    st.header("Chat with PDF :books:")

    with st.sidebar:
        st.title("Upload files")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    query = st.text_input("Ask a Question ...")

    if query:
        handle_query(query)

if __name__ == "__main__":
    main()
