import os
import requests
import warnings
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables for Azure OpenAI API
_ = load_dotenv(find_dotenv())
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY_4')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_API_ENDPOINT_4')
deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME_4')

# Set up Azure OpenAI API
openai.api_type = "azure"
openai.api_key = azure_openai_api_key
openai.api_base = azure_openai_endpoint
openai.api_version = "2023-05-15"

# Initialize FastAPI
app = FastAPI()

# Load documents from the CSV
loader = CSVLoader(file_path="gautenberg.csv", encoding="utf-8")
documents = loader.load()

# Initialize the Hugging Face embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the FAISS vector store from the loaded documents
vectorstore = FAISS.from_documents(documents, embeddings_model)

# Load the existing cleaned dataset
df = pd.read_csv('gautenberg.csv')  # Assuming this is the clean CSV you have

# Function to get book information
def get_book_info(book_title):
    book_info = df[df['Title'].str.contains(book_title, case=False, na=False)]
    if not book_info.empty:
        book_attributes = [
            f"Titre: {book_info.iloc[0]['Title']}",
            f"Auteur: {book_info.iloc[0]['Author']}",
            f"Langue: {book_info.iloc[0]['Language']}",
            f"Sujet: {book_info.iloc[0]['Subject']}",
            f"Résumé: {book_info.iloc[0]['Summary']}",
        ]
        return book_attributes, book_info.iloc[0]['Title'], book_info.iloc[0]['Ebook ID']
    else:
        return None, None, None

# Function to fetch book text from Gutenberg
def fetch_book_text(ebook_id):
    url = f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Function to ask a question about a book
def ask_book_question(question, book_attributes):
    context = "\n".join(book_attributes)
    messages = [{"role": "user", "content": f"{context}\n\nQuestion: {question}"}]
    
    # Here, use the method to ask the question to the model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Ensure this model is available
        messages=messages,
        max_tokens=200,
        temperature=0.7,
        deployment_id=deployment_name
    )
    
    return response.choices[0].message['content'].strip()

# Endpoint to ask questions about a book
@app.get("/ask_book/")
async def ask_book(title: str = Query(..., description="Titre du livre à interroger")):
    try:
        book_attributes, title, ebook_id = get_book_info(title)
        
        if book_attributes is not None:
            # Ask a question about the book's subjects
            question_subject = f"De quels sujets traite le livre '{title}' ?"
            subjects = ask_book_question(question_subject, book_attributes)

            # Ask a question to extract character names
            question_characters = f"Quels sont les noms des personnages mentionnés dans le résumé du livre '{title}' ?"
            character_names = ask_book_question(question_characters, book_attributes)

            return {
                "title": title,
                "subjects": subjects,
                "character_names": character_names
            }
        else:
            raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Endpoint to get character names
@app.get("/characters/")
async def get_characters(title: str = Query(..., description="Titre du livre")):
    try:
        book_attributes, title, ebook_id = get_book_info(title)
        
        if book_attributes is not None:
            # Extract the summary to retrieve character names
            summary = book_attributes[4]  # The summary is the fifth attribute
            question_characters = f"Quels sont les noms des personnages mentionnés dans le résumé suivant : {summary}"
            character_names = ask_book_question(question_characters, book_attributes)

            return {
                "title": title,
                "character_names": character_names
            }
        else:
            raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
