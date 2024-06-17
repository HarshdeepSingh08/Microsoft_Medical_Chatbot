from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)


index_name="medical-bot"
index = pc.Index(index_name)
#Creating Embeddings for Each of The Text Chunks & storing
vectors = []
for i, t in enumerate(text_chunks):
    embedding = embeddings.embed_documents([t.page_content])[0]
    vector = {
        "id": f"vec-{i}",
        "values": embedding,
        "metadata": {"text": t.page_content}
    }
    vectors.append(vector)

# Batch upsert vectors into Pinecone index
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(
        vectors=[
            {"id": vec["id"], "values": vec["values"], "metadata": vec["metadata"]}
            for vec in batch
        ]
    )

# Create Pinecone vector store for querying
docsearch = Pinecone(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key="text"  # Metadata key
)
print("Pinecone vector store created")
