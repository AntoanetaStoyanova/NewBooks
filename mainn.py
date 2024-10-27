from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())  # read local .env file
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY_4')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_API_ENDPOINT_4')
deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME_4')

# Initialize FastAPI app
app = FastAPI()

# Load the cleaned data
df_data = pd.read_csv('gautenberg.csv')

# Define request body
class QueryRequest(BaseModel):
    title: str
    question: str

@app.post("/ask/")
async def ask_agent(query: QueryRequest):
    # Extract the title and question from the request
    title = query.title
    question = query.question

    # Find the book by title
    book_info = df_data[df_data['Title'].str.contains(title, case=False, na=False)]

    if book_info.empty:
        raise HTTPException(status_code=404, detail="Book not found")

    # Here you would integrate your OpenAI API call
    response = generate_response(book_info, question)  # Define this function based on your logic

    return {"response": response}

# Function to generate a response using OpenAI
def generate_response(book_info, question):
    # Prepare your prompt for the OpenAI API
    prompt = f"Based on the following information about the book:\n{book_info.to_dict(orient='records')}\n\nAnswer the question: {question}"

    # Call OpenAI API
    openai.api_type = "azure"
    openai.api_key = azure_openai_api_key
    openai.api_base = azure_openai_endpoint
    openai.api_version = "2023-05-15"  # Adjust this to your API version

    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['choices'][0]['message']['content']





import os
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables for Azure OpenAI API
_ = load_dotenv(find_dotenv())  # read local .env file
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

# Load the existing cleaned dataset
df = pd.read_csv('gautenberg.csv')  # Assuming this is the clean CSV you have

# Function to interact with Azure OpenAI API
def ask_openai(prompt: str) -> str:
    try:
        response = openai.Completion.create(
            engine=deployment_name,
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to query book details by title
@app.get("/books/")
async def get_book_details(title: str = Query(..., description="Title of the book to query")):
    try:
        # Find the book based on title (case insensitive)
        book_info = df[df['Title'].str.lower().str.strip() == title.lower().strip()].iloc[0]
        
        # Convert NaN values to a placeholder ('N/A')
        book_info = book_info.fillna('N/A').replace([float('inf'), float('-inf')], 'N/A').to_dict()
        
        return book_info
    except IndexError:
        # Raise a 404 if the book is not found
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        # Catch other exceptions and raise a 500 error with the specific message
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Endpoint to interact with Azure OpenAI API
@app.post("/ask/")
async def ask_question(title: str = Query(..., description="Title of the book"), question: str = Query(..., description="Your question about the book")):
    try:
        # Retrieve the book's summary
        book_info = df[df['Title'].str.lower() == title.lower()].iloc[0]
        summary = book_info['Summary']
        
        # Create a prompt using the summary and the user's question
        prompt = f"Here is a summary of the book '{title}': {summary}\n\n{question}"
        
        # Ask Azure OpenAI API
        answer = ask_openai(prompt)
        
        return {"title": title, "question": question, "answer": answer}
    except IndexError:
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
