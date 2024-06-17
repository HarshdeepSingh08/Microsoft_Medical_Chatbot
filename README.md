# Microsoft_Medical_Chatbot
Here we have implement an end to end Medical Chatbot that takes input any medical query and response accordingly.


Tech Stack Used -:
Python 
Flask (Frontend)
Lang Chain (Generative AI Framework) 
Pinecone ( Vector DB)

How to run -:
1. Install all the dependencies from requirements.txt file.
2. Make sure to create a new .env file and keep your Pinecone API in it as PINECONE_API_KEY = "Your api key here"
3. Run python app.py

Architecture Followed -:
1. Data set import (A folder naming Data has been created and a medical book has been kept in it.)
2. Extraction of data
3. Seperated data into text chunks.
4. Converted data into vectors(embeddings)
5. Made sematantic index use above vector
6. Stored vectors in DB
   
Now comes client side -:
1. Client entered query
2. Query embedded
3. Sent to DB
4. Extracted Ranked results
5. Return the best result.

Now here you can add one more step to enhance user experience, you can use LLM model and pass ranked result into it and then it will return the response based on query.

-------------------------------THANKS----------------------------------------
