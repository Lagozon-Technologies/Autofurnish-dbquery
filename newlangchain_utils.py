import os, ast
import pandas as pd
# from google.cloud import bigquery
import datetime, yaml
from dotenv import load_dotenv
load_dotenv()
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate # type: ignore
import pandas as pd
import os

#from  langchain_openai.chat_models import with_structured_output
import json, logging

# from langchain.vectorstores import Chroma
# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI
# from langchain.embeddings import AzureOpenAIEmbeddings 
# import logging
# from logging.config import dictConfig
# logging.basicConfig(level=logging.INFO)
from logger_config import configure_logging, log_execution_time

configure_logging()
# Create main application logger
logger = logging.getLogger("app")
# # Logging configuration
# LOGGING_CONFIG = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
#         "json": {"format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'}
#     },
#     "handlers": {
#         "console": {"class": "logging.StreamHandler", "formatter": "default"},
#         "file": {"class": "logging.FileHandler", "filename": "app.log", "formatter": "json"}
#     },
#     "loggers": {
#         "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
#         "app": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False}
#     },
#     "root": {
#         "handlers": ["console"],
#         "level": "INFO"
#     }
# }
# dictConfig(LOGGING_CONFIG)
# logger = logging.getLogger("app")


AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME')
AZURE_EMBEDDING_DEPLOYMENT_NAME= os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME')
# Initialize the Azure OpenAI client
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# llm = AzureChatOpenAI(
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     azure_deployment=AZURE_DEPLOYMENT_NAME,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     temperature=0
# )
from typing import List
load_dotenv()
import csv 
from io import StringIO
#table_details_prompt = os.getenv('TABLE_DETAILS_PROMPT')
# Change if your schema is different
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")


# from langchain_community.utilities.sql_database import SQLDatabase
#from langchain.agents import create_sql_agent
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain.chains import create_sql_query_chain
# from langchain_openai import ChatOpenAI
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from langchain.memory import ChatMessageHistory
from operator import itemgetter
# from google.oauth2 import service_account
import json
from urllib.parse import quote_plus


from operator import itemgetter

# from langchain_core.output_parsers import StrOutputParser

# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI



# from table_details import get_table_details , get_tables , itemgetter ,  Table 
from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker

# import configure
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Cloud_service.json'
# db_user = os.getenv("db_user")
# db_password = os.getenv("db_password")
# db_host=os.getenv("db_host")
# #db_warehouse=os.getenv("db_warehouse")
# db_database=os.getenv("db_database")
# db_port=os.getenv("db_port")
# db_schema= os.getenv("db_schema")
db_tables =  json.loads(os.getenv("db_tables"))


SQL_DB_SERVER = os.getenv("SQL_DB_SERVER")
SQL_DB_PORT = os.getenv("SQL_DB_PORT")
SQL_DB_NAME = os.getenv("SQL_DB_NAME")
SQL_DB_USER = os.getenv("SQL_DB_USER")
SQL_DB_PASSWORD = os.getenv("SQL_DB_PASSWORD")
SQL_DB_DRIVER = os.getenv("SQL_DB_DRIVER").replace(" ", "+")  # URL encode spaces
SQL_POOL_SIZE = int(os.getenv("SQL_POOL_SIZE", 5))
SQL_MAX_OVERFLOW = int(os.getenv("SQL_MAX_OVERFLOW", 10))

SQL_DATABASE_URL = (
    f"mssql+pyodbc://{SQL_DB_USER}:{SQL_DB_PASSWORD}@{SQL_DB_SERVER}:{SQL_DB_PORT}/{SQL_DB_NAME}"
    f"?driver={SQL_DB_DRIVER}&Connection+Timeout=120"
)


# from sqlalchemy.exc import SQLAlchemyError
# def insert_feedback(department, user_query, sql_query, table_name, data, feedback_type="user not reacted", feedback="user not given feedback"):
#     engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     insert_query = text("""
#         INSERT INTO lz_feedbacks (department, user_query, sql_query, table_name, data, feedback_type, feedback)
#         VALUES (:department, :user_query, :sql_query, :table_name, :data, :feedback_type, :feedback)
#     """)

#     try:
#         session.execute(insert_query, {
#             "department": department,
#             "user_query": user_query,
#             "sql_query": sql_query,
#             "table_name": table_name,
#             "data": data,
#             "feedback_type": feedback_type,
#             "feedback": feedback
#         })
#         session.commit()
#     except Exception as e:
#         session.rollback()
#         raise e  # Propagate the exception
#     finally:
#         session.close()

# def save_votes(table_name, votes):
#     engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     execute_query = text("""
#         INSERT INTO lz_votes (table_name, upvotes, downvotes) 
#         VALUES (:table_name, :upvotes, :downvotes)
#         ON CONFLICT (table_name) 
#         DO UPDATE SET 
#             upvotes = EXCLUDED.upvotes,
#             downvotes = EXCLUDED.downvotes
#     """)

#     try:
#         session.execute(execute_query, {
#             "table_name": table_name,
#             "upvotes": votes["upvotes"],
#             "downvotes": votes["downvotes"]
#         })
#         session.commit()
#     except Exception as e:
#         session.rollback()
#         raise e  # Propagate the exception
#     finally:
#         session.close()
# # Create the connection string

# def load_votes(table_name):
#     engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     execute_query = text("""
#         SELECT upvotes, downvotes 
#         FROM lz_votes 
#         WHERE table_name = :table_name
#     """)

#     try:
#         result = session.execute(execute_query, {"table_name": table_name}).fetchone()
#         if result:
#             return {"upvotes": result[0], "downvotes": result[1]}
#         else:
#             return {"upvotes": 0, "downvotes": 0}
#     except Exception as e:
#         raise e  # Propagate the exception
#     finally:
#         session.close()
# def get_postgres_db(selected_subject, db_tables):
#     print("SELECTED SUB",selected_subject,db_tables)
#     try:
#         print(db_schema)
#         db = SQLDatabase.from_uri(
#             f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}',
#             schema=db_schema,
#             include_tables=db_tables,
#             view_support=True,
#             sample_rows_in_table_info=1,
#             lazy_table_reflection=True
#         )
#         print("Connection successful")

#     except SQLAlchemyError as e:
#           print(f"Error connecting to the database: {e}")
#     return db
def create_bigquery_uri(project_id, dataset_id):
    """Creates a BigQuery URI string."""
    return f"{project_id}.{dataset_id}"


# class BigQuerySQLDatabase(SQLDatabase):
#     def _init_(self):
#         try:
#             # Create credentials dictionary from environment variables
#             credentials_info = {
#                 "type": os.getenv('GOOGLE_CREDENTIALS_TYPE'),
#                 "project_id": os.getenv('GOOGLE_CREDENTIALS_PROJECT_ID'),
#                 "private_key_id": os.getenv('GOOGLE_CREDENTIALS_PRIVATE_KEY_ID'),
#                 "private_key":"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDln+0curmestQu\nEjLJWLY5YkCdmhlEZfWvCapN41hO6mS6nwVeYQw4ICP8ltbdsAZrsmzVgtf3GC+G\ngL99wG5WeEd/F1XPTemg8mKbMAf67nGWMc3z5yV3U4sGEnDglCVh1gHhDQC/px2K\nWopLVC/F46zQ+ERj8RjFCXExuZrzCExuFvRrT6dalDOqH8XFLeonnLoqJkPVgjvW\nfuuihW5pMiOfGyXksabfOc1GAzt4Ixbp0rsUL10ZqTPz+FOQ4WeJcs1slgRSQxHC\nmnTKx5kAT8MHEChGzhX9/BHDDzjTZL5isEybWjbKuUEcqCpc1FajFMT8NSDayifz\nEtHnxHhjAgMBAAECggEAAJYeec2r1d5/1Ttx3F3qf59TUJ/9qjwZu0SQQf2DOSvy\nuLHbYYGcUupehJ3LIBmiTIxyvEKrwibe3eJdLo5jQqZccY3OZbnN93T+8lHAMs4F\njkpRKj/WB0dF1uImLDXaTAPfM1lezVsgO71ESZ6L53fKBYrSXGLP/bVOfbcJTuwn\nFjAgNdpj2xYl+G9B+qkuNBHZ7uVnQ3w2l4zvRAWIIwtRj1qjCe7ynac9xizkrEMI\nae1WCKCZQbJGOvOl4Mu00cVvfspwzHfQZwkn+dVjN2+HNQTbzsM14CzDTmXGjD6e\n5/s3OYj7Tt4lV/PIVsf/y/zz3mtVV5D73yWQiZbiNQKBgQD9HfRuKnrKFrnyTy31\nRkSqZTfZh2smRqiR4RZssgZUCKD5GZtQ3/opWkh2HSBdQY8tLkxiu7wJ9WKmHMVV\nnUANqcBxXwsaLdMVEt4C7Y3aav8owIn+rLxD0BuQkjbX+7cx0UTnNhmg97HpYJr5\nNV+xF2LyviTemPpviWI2W6N9FwKBgQDoPXjR+L8ow0Sxhu5IjLWWp86X4KXQOCuY\n/Qbk+L3ibM8DRpgZ+nwH9zDWcGIS3Kk5t8pIQSYbthYBugkekUvtCt2dRyxIPLK9\nXnaCJFSbtpd1aaII/YF6Gp0yaap0B3+x9L4w1UrvLHK3xUcVdeb3DDCj0IVAqBg9\nqtLoktbmlQKBgQC1cTqdmh/pK79hnjbAov1n9CTD71n01yPRZrvPcRIuPP0/c4at\nw9CswgY9fQWNNAixh4XEJPVXYiq0Dt26UH3xDWVhH5Ny0bSFX7/781QDZT3Bdbu1\n7xcJuW15BgcAbnVU5cFxyIs4ozZKqDCPQh51cOFCRuFhG+IyABaCBtC8QwKBgHvw\nam0sIeBALYXMa5geN76Z+WAGTJdNkr7Hsgk6UiPnS6cE4qFikxSxL8gRG9XTGyCp\nW/OpiQva5e2v+bPteKadWN3ZoOFAO2diZT5Y4ypijHvljsrbd2DRmTjROV1IrzYq\nVeG7wozXnLVEPAZQ8JzBTafu3V4/Fwi6BGqICtXtAoGAb1QEQxRfq87q2q7DxIbm\nlxooi07TB1eevVw6r2qNRQQ5DHF+vb65Tw9ZV3E0g8/fJRD2gFC+yxgfI3iUVyyh\nIBBjKgCJOgp6zOS1L+RTNQswXxxLw+5B9j/oArHZ24j7YtKPLr+bcTNypzXn8dh8\n1U/HqFUTo1bsy8Pu35MXyco=\n-----END PRIVATE KEY-----",
#                 "client_email": os.getenv('GOOGLE_CREDENTIALS_CLIENT_EMAIL'),
#                 "client_id": os.getenv('GOOGLE_CREDENTIALS_CLIENT_ID'),
#                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
#                 "token_uri": "https://oauth2.googleapis.com/token",
#                 "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
#                 "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/bqserviceacc%40prateekproject-450509.iam.gserviceaccount.com",
#                 "universe_domain": "googleapis.com"
#             }

#             # Load credentials from dictionary
#             credentials = service_account.Credentials.from_service_account_info(
#                 credentials_info,
#                 scopes=["https://www.googleapis.com/auth/bigquery"]
#             )

#             self.project_id = credentials_info["project_id"]
#             self.client = bigquery.Client(credentials=credentials, project=self.project_id)

#         except Exception as e:
#             raise ValueError(f"Error loading credentials: {e}")
#     def run(self, command: str):
#         """Executes a SQL query and returns results as JSON."""
#         try:
#             query_job = self.client.query(command)
#             results = query_job.result()
#             return [dict(row.items()) for row in results]
#         except Exception as e:
#             return f"Error executing SQL command: {e}"

#     def get_table_names(self):
#         """Returns all available tables in the project."""
#         tables_list = []
#         datasets = list(self.client.list_datasets())
#         for dataset in datasets:
#             dataset_id = dataset.dataset_id
#             tables = self.client.list_tables(dataset_id)
#             for table in tables:
#                 tables_list.append(f"{dataset_id}.{table.table_id}")
#         return tables_list

#     def get_table_info(self, table_names=None):
#         """Returns schema information for given tables."""
#         if table_names is None:
#             table_names = self.get_table_names()

#         schema_info = ""
#         for table_name in table_names:
#             try:
#                 dataset_id, table_id = table_name.split(".")
#                 table_ref = self.client.dataset(dataset_id).table(table_id)
#                 table = self.client.get_table(table_ref)

#                 schema_info += f"\nTable: {table_name}\nColumns:\n"
#                 for column in table.schema:
#                     schema_info += f"  {column.name} ({column.field_type}) {'NULLABLE' if column.is_nullable else 'NOT NULLABLE'}\n"
#             except Exception as e:
#                 schema_info += f"Error getting schema for table {table_name}: {e}\n"

#         return schema_info



# Connection for new AZURE SQL 
# def get_sql_db():
#     try:
#         engine = create_engine(
#             SQL_DATABASE_URL,
#             pool_size=SQL_POOL_SIZE,
#             max_overflow=SQL_MAX_OVERFLOW,
#             echo=False  # Set to False in production
#         )

       

    #     Wrap the engine in a LangChain SQLDatabase object
    #     db = SQLDatabase(engine)

    #     return engine

    # except SQLAlchemyError as e:
    #     logger.error(f"Error connecting to the database: {e}")
    #     return None




def get_chain(question, selected_database, table_details, selected_business_rule,question_type,relationships, examples,final_query_instruction):
    print(" final_query_instruction  inside the get chain" ,  final_query_instruction)
    # if selected_database == 'GCP':
    #     prompt_file = "Generic_GCP_prompt.txt" if question_type == "generic" else "GCP_prompt.txt"
    # elif selected_database == 'PostgreSQL-Azure':
    #     prompt_file = "Generic_postgres_prompt.txt" if question_type == "generic" else "Postgres_prompt.txt"
    if selected_database == 'Azure SQL':
        print("prompt for azure is loaded!!")
        prompt_file = "Generic_azure_prompt.txt" if question_type == "generic" else "Azure_prompt.txt"
    
    
#     llm = AzureChatOpenAI(
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     azure_deployment=AZURE_DEPLOYMENT_NAME,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     temperature=0
# )
    def load_prompt():
        with open(prompt_file, "r", encoding="utf-8") as file:
            return file.read()

    FINAL_PROMPT = load_prompt()
    # Get the static part of the prompt
    static_prompt = FINAL_PROMPT
    # example_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         # ("human", "{input}\nSQLQuery:"),
    #         ("human", "{input}"),
    #         ("ai", "{query}"),
    #     ]
    # )
    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     example_prompt=example_prompt,
    #     example_selector=get_example_selector("sql_query_examples.json"),
    #     input_variables=["input","top_k","table_info"],
    # )
    
    Business_Glossary_Boolean = os.getenv("Business_Glossary_Boolean")

    if Business_Glossary_Boolean == "1":
        business_glossary = "## Business Glossary :\n" + get_business_glossary_text()
    else:
        business_glossary = ""
   
    
    relationships_str = relationships or "No relationships found"
    logger.info(f"submit query --> invoke_chain --> get_chain : relationship_str: {relationships_str}")

    # print("Few shot prompt : " , few_shot_prompt.invoke({{"input": "List all parts used in a particular repair order RO22A002529",
    # "top_k": "2", "table_info":""}}))
    def examples_to_str(examples):
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  input: {ex['input']}")
            lines.append(f"  query: {ex['query']}")  # Direct access to query string
            lines.append("")  # blank line between examples
        return "\n".join(lines)
    examples_str = examples_to_str(examples)
    logger.info(f"submit query --> invoke_chain --> get_chain : examples_str : {examples_str}")
    if question_type == "generic":

        final_prompt1 = static_prompt.format(table_info=table_details,  Business_Glossary = business_glossary,relationships=relationships_str, examples = examples_str,final_query_instruction=final_query_instruction)
    elif question_type =="usecase":
        final_prompt1 = static_prompt.format(table_info=table_details, Business_Rule = selected_business_rule, Business_Glossary = business_glossary, relationships=relationships_str, examples = examples_str,final_query_instruction=final_query_instruction)
    final_prompt = final_prompt1
    # print("prompt here: ", final_prompt)
    # if selected_database=="GCP":
    #         db = BigQuerySQLDatabase()
    # elif selected_database=="PostgreSQL-Azure":
    #     db = get_postgres_db(selected_subject, db_tables)
    # print("Generate Query Starting")

    try:
   
        response = azure_openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            store = True,
            messages=[
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0,  # Lower temperature for more predictable, structured output
        response_format={"type": "json_object"}  # This is the key parameter!
        )

    # The response content will be a JSON string
        response_content = response.choices[0].message.content
        logger.info(f"submit query --> invoke_chain --> get_chain : response from azure LLM: {response_content}")

        # Parse the guaranteed JSON string into a Python dictionary
        json_output = json.loads(response_content)

        # Now you can safely access the keys
        # print("--- LLM Output (Parsed) ---")
        # print(f"Description: {json_output.get('description')}")

        SQL_Statement = json_output.get('query')
        # logger.info(f"Query: {json_output.get('query')}")
        logger.info(f"submit query --> invoke_chain --> get_chain :LLM SQL Generation Error: {json_output.get('error')}")
        logger.info(f"submit query --> invoke_chain --> get_chain : Description: {json_output.get('description')}")
    except Exception as e:
        logger.error(f"Error in submit query --> invoke_chain --> get_chain : {e}")

    #     final_prompt=final_prompt2    
    # generate_query = create_sql_query_chain(llm, db, final_prompt)
    # SQL_Statement = generate_query.invoke({"question": question, "messages": _messages})

    # # Override QuerySQLDataBaseTool validation
    # class CustomQuerySQLDatabaseTool(QuerySQLDataBaseTool):
    #     def _init_(self, db):
    #         if not isinstance(db, SQLDatabase):
    #             raise ValueError("db must be an instance of SQLDatabase")
    #         super()._init_(db=db)

    # execute_query = CustomQuerySQLDatabaseTool(db=db)
    
    # chain = (
    #     RunnablePassthrough.assign(table_names_to_use=lambda _: ["MH_RO_HDR_DETAILS", "MH_RO_PARTS", "verbatim_cmm_aruna", "srv_mst_dim_model_master", "ddp_ad_ai_final_dimension", "MH_RO_LABOUR"]) |
    #     RunnablePassthrough.assign(query=generate_query).assign(
    #         result=itemgetter("query")
    #     )
    # )
    
        
    
    # return chain,  SQL_Statement, db,final_prompt1
    # logger.info(f" submit query --> invoke_chain --> get_chain, Generated SQL Statement : {SQL_Statement}")

    return json_output, final_prompt1

def invoke_chain(db,question, messages, selected_model, selected_subject, selected_database, table_info, selected_business_rule, question_type, relationships, examples,final_query_instruction):
    logger.info(f"submit query --> invoke_chain , parameters are: {question, messages, selected_model, selected_subject, selected_database}")
    response = None
    SQL_Statement = None
    final_prompt = None
    try:
        # history = create_history(messages)
        json_output, final_prompt = get_chain(
            question, 
            selected_database, table_info, selected_business_rule, question_type, relationships, examples,final_query_instruction
        )  ##we get query output from get chain
        SQL_Statement = json_output["query"]
        description = json_output["description"]
        print("This is our description", description)

        # SQL_Statement = SQL_Statement.replace("SQL Query:", "").strip()

        # response = chain.invoke({
        #     "question": question,
        #     "top_k": 1,
        #     "messages": history.messages,
        #     "table_details": table_info
        # })
        # print("Question:", question)
        # print("Response:", json_output)

        tables_data = {}
        query = SQL_Statement
        print(f"submit query --> invoke_chain: Executing SQL Query: {query}")
        # if selected_database == "GCP":
        #     rows = db.query(query).result()
        #     result_json = [dict(row) for row in rows]
        #     logger.info(f"submit query --> invoke_chain : result after executing : \n{result_json}")
        #     df = pd.DataFrame(result_json)
        #     tables_data["Table data"] = df
            # break
        # elif selected_database == "PostgreSQL-Azure":
        #     alchemyEngine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
        #     with alchemyEngine.connect() as conn:
        #         df = pd.read_sql(sql=query, con=conn.connection)
        #     tables_data[table] = df
        #     print(table)
            # break 
        if selected_database == "Azure SQL":
            # db = get_sql_db()
            logger.info("in if of azure sql")
            result = db.execute(text(query))
            logger.info(f"submit query --> invoke_chain : result after executing \n{query} : \n{result}")
            try:
                rows = result.fetchall()
                columns = result.keys()
                df = pd.DataFrame(rows, columns=columns)
                tables_data["Table data"] = df
            except Exception as e:
                logger.error(f"submit query --> invoke_chain : table data has some issues.{e}")
        # Include SQL_Statement in the return tuple
        return json_output, db_tables, tables_data, final_prompt,description

    except Exception as e:
        print("submit query --> invoke_chain :", e)
        # Return whatever response was generated, or None if none was generated
        # Also return SQL_Statement and final_prompt if available
        return response, [], {}, final_prompt,None

# def create_history(messages):
#     history = ChatMessageHistory()
#     for message in messages:
#         if message["role"] == "user":
#             history.add_user_message(message["content"])
#         else:
#             history.add_ai_message(message["content"])
#     return history

def escape_single_quotes(input_string):
    return input_string.replace("'", "''")

glossary = False #boolean for glossary function that is passed in prompt for business words and their explanations
def get_business_glossary_text():
    # Read the business glossary CSV
    glossary_df = pd.read_csv('table_files/business_glossary.csv')
    
    # Format each row as "Business Word: Explanation"
    glossary_lines = [
        f"{row['Business Word']}: {row['Explanation']}"
        for _, row in glossary_df.iterrows()
    ]
    
    # Join all lines with newline characters
    glossary_text = '\n'.join(glossary_lines)
    return glossary_text

def get_key_parameters():
    params = os.getenv('key_parameters')
    # Split by comma and strip whitespace
    return [p.strip() for p in params.split(',') if p.strip()]

def read_defaults(csv_content):
    f = StringIO(csv_content)
    reader = csv.DictReader(f)
    result = {}
    for row in reader:
        key = row['default']
        value = row['value']
        if value == 'Current_date':
            value = datetime.now().strftime('%Y-%m-%d')
        result[key] = value
    return result
def intent_classification(user_query):
    user_query_lower = user_query.lower()
    matched_tables = set()
    detected_intent = None

    with open('table_files/Intentclass.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            keywords = [k.strip().lower() for k in row['Keywords'].replace(';', ',').split(',')]
            if any(keyword and keyword in user_query_lower for keyword in keywords):
                detected_intent = row['Intent']
                table_names = [t.strip() for t in row['tables'].split(';') if t.strip()]
                matched_tables.update(table_names)
                break  # stop after the first matching intent (optional)

    if detected_intent and matched_tables:
        logger.info(f"submit query --> intent_classification,usecase, Returned intent:, {detected_intent}")
        logger.info(f"submit query --> intent_classification,usecase, Returned tables from intent: {matched_tables}")
        return {
            "intent": detected_intent,
            "tables": list(matched_tables)
        }
    return False

import ast

def get_business_rule(intent, file_path='business_rules.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            # Normalize all keys to lowercase for case-insensitive lookup
            business_rules = {k.lower(): v for k, v in ast.literal_eval(file_content).items()}
    except Exception as e:
        return f"submit query --> get_business_rule, Error reading business rules file: {e}"

    key = intent.lower().strip()
    rule = business_rules.get(key)

    return rule if rule else f"submit query --> get_business_rule, No business rule defined for intent: {intent}"

# def get_example_selector(json_file_path: str):
#     """
#     Returns a SemanticSimilarityExampleSelector initialized with examples from a JSON file.
    
#     Args:
#         json_file_path (str): Path to the JSON file containing examples
        
#     Returns:
#         SemanticSimilarityExampleSelector: Selector configured with examples
#     """
#     # Load examples from JSON file
#     with open(json_file_path, 'r', encoding='utf-8') as file:
#         examples = json.load(file)
    
#     # Validate examples structure
#     if not isinstance(examples, list):
#         raise ValueError("JSON file should contain a list of examples")
#     if len(examples) == 0:
#         raise ValueError("No examples found in JSON file")
#     if not all(isinstance(example, dict) and 'input' in example and 'query' in example for example in examples):
#         raise ValueError("Each example should be a dictionary with 'input' and 'query' keys")
    
#     # Create Azure OpenAI embeddings instance
#     azure_embeddings = AzureOpenAIEmbeddings(
#         azure_deployment= AZURE_EMBEDDING_DEPLOYMENT_NAME,  # Your embedding model deployment name
#         openai_api_version=AZURE_OPENAI_API_VERSION,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_key=AZURE_OPENAI_API_KEY,
#     )
    
#     # Create example selector with Azure embeddings
#     example_selector = SemanticSimilarityExampleSelector.from_examples(
#         examples,
#         azure_embeddings,  # Use Azure embeddings instead of OpenAIEmbeddings
#         Chroma,
#         k=3,
#         input_keys=["input"],
#     )
    
#     return example_selector

def find_relationships_for_tables(table_names, yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        relations_data = yaml.safe_load(f)
    filtered = []
    for rel in relations_data.get("RELATIONSHIPS", []):
        if rel.get("left_table") in table_names or rel.get("right_table") in table_names:
            filtered.append(rel)
    return yaml.dump({"RELATIONSHIPS": filtered}, sort_keys=False, allow_unicode=True)