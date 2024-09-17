import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio
 
# Initialize FastAPI app
app = FastAPI()
 
# Load environment variables from .env file
load_dotenv()
 
# Define the QuestionType Enum
class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_false = "true_or_false"
    estimation = "estimation"
 
# Define the Ask model
class Ask(BaseModel):
    question: str
    type: QuestionType
    correlationToken: str | None = None
 
# Define the Answer model
class Answer(BaseModel):
    answer: str
    correlationToken: str | None = None
    promptTokensUsed: int | None = None
    completionTokensUsed: int | None = None
 
# Set up Azure OpenAI client - re-use it for efficiency
client: AzureOpenAI = None
 
if "AZURE_OPENAI_API_KEY" in os.environ:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
 
deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
 
# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello, welcome to the Q&A API!"}
 
# Endpoint to ask a question
@app.post("/ask", summary="Ask a question", operation_id="ask") 
async def ask_question(ask: Ask):
    if not client:
        raise HTTPException(status_code=500, detail="Azure OpenAI client not initialized")
    # Send a completion call to generate an answer
    print('Sending a request to Azure OpenAI')
    try:
        response = await asyncio.to_thread(client.chat.completions.create,
                                           model=deployment_name,
                                           messages=[{"role": "user", "content": ask.question}]
                                           )
        # Create the Answer model from the response
        answer = Answer(
            answer=response.choices[0].message.content,
            correlationToken=ask.correlationToken,
            promptTokensUsed=response.usage.prompt_tokens,
            completionTokensUsed=response.usage.completion_tokens
        )
        return answer
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))