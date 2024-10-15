import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_openai.embeddings import OpenAIEmbeddings
import uuid 


# Initialize the app ============================
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

pineKey = st.secrets["PINECONE_API_KEY"]
pineEnv = st.secrets["PINECONE_ENVIRONMENT"]
pineInd = st.secrets["PINECONE_INDEX_NAME2"]
pinemod = "text-embedding-3-small"


class Document:
    def __init__(self, text, metadata=None, doc_id=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}
        self.id = doc_id if doc_id is not None else str(uuid.uuid4())  # Uniek ID genereren


def main():
    doc_db = embedding_db()
    print(doc_db)
    print("Bestanden weggeschreven naar PineCone DB")


def embedding_db():
    embeddings = OpenAIEmbeddings(model=pinemod)  
    pc = Pinecone(api_key=pineKey)

    if pineInd not in pc.list_indexes().names():
        pc.create_index(
            name=pineInd,
            dimension=1536,  
            metric='cosine',  
            spec=ServerlessSpec(
                cloud='gcp',  
                region=pineEnv  
            )
        )

    docs_split = load_embeddings_from_dir()

    doc_db = LangChainPinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name=pineInd
    )
    
    return doc_db


def load_embeddings_from_dir():
    directory = './files/'
    documents = []  
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                parts = content.split('\n\n')
                for part in parts:
                    documents.append(Document(part))
    print(f"Aantal gesplitste documenten: {len(documents)}")
    return documents


# Start the app ===============================
if __name__ == "__main__":
    main()
