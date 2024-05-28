import json
import os
import time

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.confluence import ConfluenceLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders.text import TextLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from db_configs import index_name, embedding, EMBEDDING_TO_DIMENSION, ERROR_COLS, SOLUTION_COLS

load_dotenv()
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model=embedding, dimensions=EMBEDDING_TO_DIMENSION[embedding])
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimensions,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore

def load_data():
    vectorstore = get_vectorstore()
    json_data = json.load(open('/Users/avinoam/dev/something-went-wrong/resources/data.json'))
    error_keys = []
    solution_metadatas = []
    for data in json_data:
        if not data.get("Solutions"):
            continue
        if not any((data.get(c) for c in ERROR_COLS)):
            continue
        key_error = ""
        for c in ERROR_COLS:
            if data.get(c):
                key_error += f"{c}: {data.get(c)}\n"
        metadata_dict = {}
        for c in SOLUTION_COLS:
            if data.get(c):
                metadata_dict[c] = data.get(c)
        error_keys.append(key_error)
        solution_metadatas.append(metadata_dict)

    vectorstore.add_texts(texts=error_keys, metadatas=solution_metadatas)

def get_context_from_db(query):
    vectorstore = get_vectorstore()
    res = vectorstore.similarity_search(query, k=1)
    res_str = ', '.join(f"{key}: {value}" for key, value in res[0].metadata.items())
    return res_str
