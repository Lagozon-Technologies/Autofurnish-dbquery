import chromadb
from chromadb.utils import embedding_functions
import json
import os
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from logger_config import configure_logging, log_execution_time
configure_logging()
# Create main application logger
logger = logging.getLogger("app")

# Load both types of examples
with open("sql_query_examples_generic.json", encoding="utf-8") as f:
    generic_examples = json.load(f)

with open("sql_query_examples_usecase.json", encoding="utf-8") as f:
    usecase_examples = json.load(f)




# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION')
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME')
CHROMA_DB_PATH = os.environ.get('Chroma_Query_Examples')

# Initialize embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_type="azure",
    api_version=AZURE_OPENAI_API_VERSION,
    model_name=AZURE_EMBEDDING_DEPLOYMENT_NAME
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def prepare_ingest(items):
    with log_execution_time("Submit Endpoint in main"):
        try:

            inputs = [item['input'] for item in items]
            queries = [item['query'] for item in items]
            return inputs, queries
        except Exception as e:
            logger.error(f"Error in prepare_ingest, {e}")
def ingest_examples(examples, collection_name):
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    inputs = []
    queries = []
    metadatas = []  # We'll build complete metadata dictionaries
    
    # Check if this is the new format with intents (usecase)
    if isinstance(examples, list) and len(examples) > 0 and "intent" in examples[0]:
        logger.info(f"Ingesting {len(examples)} intent groups into {collection_name}")
        for intent_group in examples:
            for example in intent_group["examples"]:
                inputs.append(example["input"])
                queries.append(example["query"])
                # For usecase examples, include intent in metadata
                metadatas.append({"query": example["query"], "intent": intent_group["intent"]})
    else:
        logger.info(f"Ingesting {len(examples)} generic examples into {collection_name}")
        for example in examples:
            inputs.append(example["input"])
            queries.append(example["query"])
            # For generic examples, only include query in metadata
            metadatas.append({"query": example["query"]})  # No intent field
    
    if not inputs:
        logger.warning(f"No examples found to ingest for {collection_name}")
        return collection
    
    ids = [f"{collection_name}pair{i}" for i in range(len(inputs))]
    
    collection.upsert(
        ids=ids,
        documents=inputs,
        metadatas=metadatas  # Use our prepared metadata list
    )
    logger.info(f"Successfully ingested {len(inputs)} examples into {collection_name}")
    return collection

# Ingest both types of examples
generic_collection = ingest_examples(generic_examples, "generic_examples")
usecase_collection = ingest_examples(usecase_examples, "usecase_examples")