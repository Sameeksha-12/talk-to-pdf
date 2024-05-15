import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# side bar contents
with st.sidebar:
    st.title('Talk to PDFs')
    st.markdown('''
    ## About
    This app is an LLM chatbot built using:
    - Streamlit
    - Langchain
    - OpenAI            
    ''')
    add_vertical_space(5)


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
def main():
    st.header("Chat with PDF 💬")

    # uploading the file
    pdf = st.file_uploader("Upload your PDF",type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write(chunks)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f) 
            st.write('Embeddings loaded from the disk')
        else:
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        #Accept user questions/query
        query = st.text_input("Ask questions about your PDF File:")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()

