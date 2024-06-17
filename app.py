from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec

app = Flask(__name__)

load_dotenv()

# Pinecone API key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Define the index name and create Pinecone index if not exists
index_name = "medical-bot"


# Connect to the index
index = pc.Index(index_name)
print("Pinecone index initialized")

# Download embeddings model
embeddings = download_hugging_face_embeddings()

@app.route("/")
def index_page():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print("User input:", input_text)

    try:
        # Step 1: Generate embedding for the user query
        query_embedding = embeddings.embed_query(input_text)
        
        # Step 2: Query Pinecone to find the most similar vector
        query_result = index.query(vector=query_embedding, top_k=1, include_metadata=True)
        
        # Step 3: Extract the best matching document's metadata
        best_match = query_result['matches'][0]['metadata']
        response = best_match['text']
        
        print("Response:", response)
        
        # Return JSON response with the text content
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "An error occurred. Please try again."})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
