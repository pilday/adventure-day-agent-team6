import os
import json
import requests
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import redis
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from azure.search.documents.models import (
    VectorizedQuery
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

app = FastAPI()

load_dotenv()


def get_embedding(text, model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_or_false = "true_or_false"
    popular_choice = "popular_choice"
    estimation = "estimation"

class Ask(BaseModel):
    question: str | None = None
    type: QuestionType
    correlationToken: str | None = None

class Answer(BaseModel):
    answer: str
    correlationToken: str | None = None
    promptTokensUsed: int | None = None
    completionTokensUsed: int | None = None

client: AzureOpenAI

if "AZURE_OPENAI_API_KEY" in os.environ:
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
index_name = "movies-semantic-index"
service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_COMPLETION_MODEL")

# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')
 
# Connect to the Redis server
#conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)
 


#if conn.ping():
#    print("Connected to Redis")

def get_num_tokens_from_string(string: str, encoding_name: str='p50k_base') -> int:
    """Returns the number of tokens in a text by a given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

@app.get("/")
async def root():
    return {"message": "Hello Smorgs"}

@app.post("/ask", summary="Ask a question", operation_id="ask") 
async def ask_question(ask: Ask):
    """
    Ask a question
    """

    print (ask.question)
    question = ask.question
    index_name = "question-semantic-index"

    # create new searchclient using our new index for the questions
    credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]) if len(os.environ["AZURE_AI_SEARCH_KEY"]) > 0 else DefaultAzureCredential()
    search_client = SearchClient(
        endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"], 
        index_name=index_name,
        credential=credential
    )

    
    # check if the question &  answer is in the cache already
    # TODO    
    vector = VectorizedQuery(vector=get_embedding(question), k_nearest_neighbors=5, fields="vector")

    # create search client to retrieve movies from the vector store
    found_questions = list(search_client.search(
        search_text=None,
        query_type="semantic",
        semantic_configuration_name="question-semantic-config",
        vector_queries=[vector],
        select=["question", "answer"],
        top=5
    ))

    for q in found_questions:
        print(q["answer"])

    if(len(found_questions)>0):
        print ("Found a match in the cache.")
        # put the new question & answer in the cache as well
        # TODO  
        docIdCount:int = search_client.get_document_count()  +1 
        newQuestion = { "id": str(docIdCount), "question": question, "answer": found_questions[0]["answer"], "vector": get_embedding(question) }
        search_client.upload_documents([newQuestion])       
        # return the answer        
        # TODO        
        answer = newQuestion["answer"]


        token = number_of_tokens=get_num_tokens_from_string(question)
        prompt_tokens = token
        completion_tokens = token
    
    else:
        print("No match found in the cache.")        
        
        #   reach out to the llm to get the answer. 
        print('Sending a request to LLM')
        start_phrase = ask.question
        messages=  [{"role" : "assistant", "content" : start_phrase},
                     { "role" : "system", "content" : "Answer this question with a very short answer. Don't answer with a full sentence, and do not format the answer."}]
        
        response = client.chat.completions.create(
             model = deployment_name,
             messages =messages,
        )
        answer = Answer(answer=response.choices[0].message.content)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        #  put the new question & answer in the cache as wel
        docIdCount:int = search_client.get_document_count()  +1 
        
        # TODO
        newQuestion = {"id": str(docIdCount), "question": question, "answer": answer.answer, "vector": get_embedding(question)  }
        search_client.upload_documents([newQuestion]) 

        print ("Added a new answer and question to the cache: " + answer.answer + "in position" + str(docIdCount))
        

    #####\n",
    # implement cached rag flow here\n",
    ######\n",
    
    answer = Answer(answer=answer)
    answer.correlationToken = ask.correlationToken
    answer.promptTokensUsed = prompt_tokens
    answer.completionTokensUsed = completion_tokens

    return answer
