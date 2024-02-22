import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

load_dotenv()

st.title('News Research Tool')
st.sidebar.title('News Articles URLs')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL  {i+1}')
    urls.append(url)

process_url = st.sidebar.button('Process URLs')

main_placeholder = st.empty()

file_path = 'faiss_openai.pkl'  # Default value

if process_url:
    # Load Data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # Split Data
    Split_text = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = Split_text.split_documents(data)
    # Create Embeddings
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    # Save in pickle format
    with open(file_path, 'wb') as f:
        pickle.dump(vectors, f)

query = main_placeholder.text_input('Question : ')
if query:
    if os.path.exists(file_path):
        with (open(file_path, 'rb') as f):
            vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
            results = chain({'question':query}, return_only_outputs=True)
            st.header('Answer :')
            st.write(results['answer'])

            #Display source
            source = results.get('sources','')
            if source :
                st.subheader('Sources:')
                sources_list = source.split('\n')
                for source in sources_list :
                    st.write(source)