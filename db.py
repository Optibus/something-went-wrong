import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "sww-demo"

if index_name not in pc.list_indexes():
    pc.create_index(
        name="sww-demo",
        dimension=8, # Replace with your model dimensions
        metric="euclidean", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)