# from openai import OpenAI
# import json
import os 
# import openai
from openai import AzureOpenAI
from IngestExamples import generic_collection, usecase_collection
import os
from logger_config import configure_logging, log_execution_time
import logging

configure_logging()
# Create main application logger
logger = logging.getLogger("app")
# from dotenv import load_dotenv
# Initialize logger


# load_dotenv()
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME')
AZURE_EMBEDDING_DEPLOYMENT_NAME= os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME')


# openai.api_type = "azure"
# openai.api_key = AZURE_OPENAI_API_KEY
# openai.api_base = AZURE_OPENAI_ENDPOINT  
# openai.api_version = AZURE_OPENAI_API_VERSION  
AZURE_EMBEDDING_DEPLOYMENT = AZURE_EMBEDDING_DEPLOYMENT_NAME

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def embed_query(text):
    response = client.embeddings.create(
        input=[text],
        model=os.environ['AZURE_EMBEDDING_DEPLOYMENT_NAME'],
    )
    return response.data[0].embedding

def get_examples(query: str, question_type: str, intent: str = None):
    with log_execution_time("Get_examples: "):
        try:

            if question_type not in ["generic", "usecase"]:
                raise ValueError("question_type must be either 'generic' or 'usecase'")
            
            try:
                collection = generic_collection if question_type == "generic" else usecase_collection
                query_embedding = embed_query(query)
                
                # Build the filter if intent is provided (only for usecase)
                where_filter = None
                if question_type == "usecase" and intent:
                    where_filter = {"intent": {"$eq": intent}}
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
                
                examples = []
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    # Handle both metadata formats:
                    # 1. For generic examples: meta is {"query": "..."}
                    # 2. For usecase examples: meta is {"query": "...", "intent": "..."}
                    query_text = meta.get('query', '')  # Safely get query text
                    
                    examples.append({
                        'input': doc,
                        'query': query_text
                    })
                
                return examples
            
            except Exception as e:
                logger.error(f"Error in get_examples: {str(e)}", exc_info=True)
                return []
        except Exception as e:
            logger.error(f"Error in get_examples in SM_Examples.py: {e}")
# print(get_examples("Provide the list of Top 10 consumed OEM part under  Repair  from 01-Jan-2024 to 31-dec-2024","generic",intent="LABOUR_RUNNING_REPAIR"))