from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, File,Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
# from langchain_openai import ChatOpenAI
import plotly.graph_objects as go, plotly.express as px, decimal, numpy as np
import openai, yaml, os, csv,pandas as pd, base64, uuid
# from pydantic import BaseModel
from io import BytesIO, StringIO
# from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import Field, BaseModel
# from langchain_openai import ChatOpenAI
from newlangchain_utils import *
from dotenv import load_dotenv
# from state import session_state, session_lock
from typing import Optional, List, Dict
from starlette.middleware.sessions import SessionMiddleware  # Correct import
# from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from starlette.middleware.base import BaseHTTPMiddleware
import logging, time
from decimal import Decimal

# import automotive_wordcloud_analysis as awa
import  asyncio
from wordcloud import WordCloud
from table_details import get_table_details, get_table_metadata  # Importing the function
from openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI
from SM_examples import get_examples
# Configure logging
# logging.basicConfig(level=logging.INFO)
from logger_config import configure_logging, log_execution_time
from google.cloud import bigquery
from google.oauth2 import service_account

from sqlalchemy import create_engine

# from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dependencies import  get_db
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import secrets

DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "100/minute")
SENSITIVE_RATE_LIMIT = os.getenv("SENSITIVE_RATE_LIMIT", "11/minute") 
HEALTH_CHECK_LIMIT = os.getenv("HEALTH_CHECK_LIMIT", "10/second")
FILE_UPLOAD_LIMIT = os.getenv("FILE_UPLOAD_LIMIT", "5/minute")
SQL_POOL_SIZE=os.getenv("SQL_POOL_SIZE")
SQL_MAX_OVERFLOW=os.getenv("SQL_MAX_OVERFLOW")
# import redis
configure_logging()
# Create main application logger
logger = logging.getLogger("app")
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log the request details
        logging.info(f"Request: {request.method} {request.url}")
        
        # Call the next middleware or endpoint
        response = await call_next(request)
        
        # Log the response details
        logging.info(f"Response status: {response.status_code}")
        
        return response

load_dotenv()  # Load environment variables from .env file
# --- Helper function: the actual DB ping ---
keep_alive_interval = os.getenv("keep_alive_interval")
Query_Record_Size_Boolean = os.getenv("Query_Record_Size_Boolean")
Query_record_size = os.getenv("Query_Record_Size")

if Query_Record_Size_Boolean == "1":
    final_query_instruction = (
        f"- Always apply TOP {Query_record_size} in the SELECT clause to limit results "
        f"unless a lower limit (like TOP 5, TOP 10, etc.) is explicitly specified by the user."
    )

else:
    final_query_instruction = ""


print("final_query_instruction",final_query_instruction)

def run_keepalive_query(engine):
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logging.info("Keep-alive DB ping successful.")

# --- The async keep-alive task ---
async def keep_all_connections_alive(engine, pool_size, interval=keep_alive_interval):
    logging.info("Keep-alive background task started.")

    while True:
        for _ in range(pool_size):
            try:
                logging.info(f"Pinging DB connection {_+1}/{pool_size}.")
                await asyncio.to_thread(run_keepalive_query, engine)
            except Exception as e:
                logging.warning(f"Keep-alive ping failed: {e}")
            await asyncio.sleep(1)  # Small pause between pings
        await asyncio.sleep(interval)
# pool_size=int(SQL_POOL_SIZE)
# max_overflow=int(SQL_MAX_OVERFLOW)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize credentials
    engine = create_engine(
      

        SQL_DATABASE_URL,
        pool_size=int(SQL_POOL_SIZE),
        max_overflow=int(SQL_MAX_OVERFLOW),
        echo=False,
        pool_recycle=1200,  # Recycle every 20 minutes
        pool_pre_ping=True  # Validate connection before using
    )
    app.state.engine = engine
    app.state.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Initialize BigQuery client with explicit project
    
    
    # Set default dataset reference
    try:
        yield
    finally:
        # No explicit cleanup needed for BigQuery client
        engine.dispose()
    # Azure OpenAI LLM
    # app.state.azure_openai_client = AzureOpenAI(
    #     azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
    #     api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    # )

    # Azure SQL DB
    # engine = create_engine(
      

    #     SQL_DATABASE_URL,
    #     pool_size=int(SQL_POOL_SIZE),
    #     max_overflow=int(SQL_MAX_OVERFLOW),
    #     echo=False,
    #     pool_recycle=1200,  # Recycle every 20 minutes
    #     pool_pre_ping=True  # Validate connection before using
    # )
    # app.state.engine = engine
    # app.state.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    #  # Start the keep-alive background task
    # keepalive_task = asyncio.create_task(
    #     keep_all_connections_alive(engine, pool_size, interval=600)
    # )

    # # --- RETRY LOGIC FOR WARM-UP ---
    # max_attempts = 10
    # for attempt in range(1, max_attempts + 1):
    #     try:
    #         with engine.connect() as connection:
    #             connection.execute(text("SELECT 1"))
    #         logging.info("Warm-up DB connection successful.")
    #         break  # Success, exit the retry loop
    #     except Exception as e:
    #         logging.warning(f"Warm-up DB connection attempt {attempt} failed: {e}")
    #         if attempt < max_attempts:
    #             time.sleep(2 ** attempt)  # Exponential backoff
    #         else:
    #             logging.error("All warm-up DB connection attempts failed.")
    #             # Optionally: raise or exit if you want to fail fast
    # --- END RETRY LOGIC ---


    # Azure Redis Cache (async)
    # app.state.redis_client = redis.Redis(
    #     host=os.environ["REDIS_HOST"],
    #     port=int(os.environ.get("REDIS_PORT", 6380)),
    #     password=os.environ["REDIS_KEY"],
    #     ssl=True,
    #     decode_responses=True,
    #     max_connections=20
    # )
    # try:
    #     yield
    # # # app.state.redis_client.close()
    # finally:
    #     # On shutdown: cancel the keep-alive task and clean up
    #     keepalive_task.cancel()
    #     try:
    #         await keepalive_task
    #     except asyncio.CancelledError:
    #         pass
    #     engine.dispose()

# Remove the storage parameter and simplified initialization
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.add_middleware(SlowAPIMiddleware) 
secret_key = secrets.token_hex(32) 
app.add_middleware(SessionMiddleware, secret_key=secret_key)
app.add_middleware(LoggingMiddleware)
# Set up static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')


app.state.limiter = limiter
# Custom rate limit handler
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Too many requests. Limit: {exc.detail}",
            "retry_after": str(exc.detail)
        },
        headers={"Retry-After": str(exc.detail)}
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)



# Initialize the BlobServiceClient
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    logger.info("Blob service client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing BlobServiceClient: {e}")
    # Handle the error appropriately, possibly exiting the application
    raise  # Re-raise the exception to prevent the app from starting
from pydantic import BaseModel
# class ChartRequest(BaseModel):
#     """
#     Pydantic model for chart generation requests.
#     """
#     table_name: str
#     x_axis: str
#     y_axis: str
#     chart_type: str

#     class Config:  # This ensures compatibility with FastAPI
#         json_schema_extra = {
#             "example": {
#                 "table_name": "example_table",
#                 "x_axis": "column1",
#                 "y_axis": "column2",
#                 "chart_type": "Line Chart"
#             }
#         }

# Initialize OpenAI API key and model
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION')
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME')

# Initialize the Azure OpenAI client
azure_openai_client = AzureOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,

    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# llm = AzureOpenAI(
#     api_version=AZURE_OPENAI_API_VERSION,
#     azure_deployment=AZURE_DEPLOYMENT_NAME,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
# )

databases = ["Azure SQL"]
# question_dropdown = os.getenv('Question_dropdown')

import datetime

def convert_dates(obj):
    if isinstance(obj, dict):
        return {k: convert_dates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates(item) for item in obj]
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float for JSON serialization
    else:
        return obj
class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

def download_as_excel(data: pd.DataFrame, filename: str = "data.xlsx"):
    """
    Converts a Pandas DataFrame to an Excel file and returns it as a stream.

    Args:
        data (pd.DataFrame): The DataFrame to convert.
        filename (str): The name of the Excel file.  Defaults to "data.xlsx".

    Returns:
        BytesIO:  A BytesIO object containing the Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)  # Reset the pointer to the beginning of the stream
    return output

# @app.get("/health")
# def health_check():
#     try:
#         with app.state.engine.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         return {"status": "ok"}
#     except Exception:
#         return {"status": "db_error"}, 503

@app.get("/get_prompt")
@limiter.limit(SENSITIVE_RATE_LIMIT)  # Stricter limit for sensitive endpoint
async def get_prompt(request: Request, type: str):
    if type == "interpretation":
        filename = "chatbot_prompt.yaml"
    elif type == "langchain":
        filename = "final_prompt.txt"
    else:
        return "Invalid prompt type", 400
    try:
        with open(filename, "r",encoding='utf-8') as f:
            prompt = f.read()
        return prompt
    except FileNotFoundError:
        return "Prompt file not found", 404
def create_gauge_chart_json(title, value, min_val=0, max_val=100, color="blue", subtext="%"):
    """
    Creates a gauge chart using Plotly and returns it as a JSON string.

    Args:
        title (str): The title of the chart.
        value (float): The value to display on the gauge.
        min_val (int): The minimum value of the gauge.  Defaults to 0.
        max_val (int): The maximum value of the gauge.  Defaults to 100.
        color (str): The color of the gauge.  Defaults to "blue".
        subtext (str): The subtext to display below the value. Defaults to "%".

    Returns:
        str: A JSON string representation of the gauge chart.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': 'black'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 1},
            'bgcolor': "white",
            'borderwidth': 0.7,
            'bordercolor': "black",

            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': subtext, 'font': {'size': 16, 'color': 'gray'}}
    ))

    # Adjust the layout to prevent cropping
    fig.update_layout(
        width=350,  # Increased width
        height=350,  # Increased height
        margin=dict(
            t=50,  # Top margin
            b=50,  # Bottom margin
            l=50,  # Left margin
            r=50   # Right margin
        )

    )
    return fig.to_json()  # Return JSON instead of an image

class QueryInput(BaseModel):
    """
    Pydantic model for user query input.
    """
    query: str

@app.post("/add_to_faqs")
async def add_to_faqs(data: QueryInput, subject:str, request:Request):
    """
    Adds a user query to the FAQ CSV file on Azure Blob Storage.

    Args:
        data (QueryInput): The user query.

    Returns:
        JSONResponse: A JSON response indicating success or failure.
    """
    print(f"subject: {subject}")
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Invalid query!")
    question_type = request.session.get('current_question_type')

    if question_type == 'generic':
        blob_name = f'table_files/Azure-SQL-DB_questions_generic.csv'
    elif question_type == "usecase":
        blob_name = f'table_files/Azure-SQL-DB_questions.csv'
    try:
        # Get the blob client
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)

        try:
            # Download the blob content
            blob_content = blob_client.download_blob().content_as_text()
        except ResourceNotFoundError:
            # If the blob doesn't exist, create a new one with a header if needed
            blob_content = "question\n"  # Replace with your actual header

        # Append the new query to the existing CSV content
        updated_csv_content = blob_content + f"{query}\n"  # Append new query

        # Upload the updated CSV content back to Azure Blob Storage
        blob_client.upload_blob(updated_csv_content.encode('utf-8'), overwrite=True)

        return {"message": "Query added to FAQs successfully and uploaded to Azure Blob Storage!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def generate_chart_figure(data_df: pd.DataFrame, x_axis: str, y_axis: str, chart_type: str):
    """
    Generates a Plotly figure based on the specified chart type.
    Includes support for Word Cloud visualization.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the data.
        x_axis (str): The column name for the x-axis.
        y_axis (str): The column name for the y-axis.
        chart_type (str): The type of chart to generate.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure, or None if the chart type is unsupported.
    """
    fig = None
    try:
        if chart_type == "Line Chart":
            fig = px.line(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Bar Chart":
            fig = px.bar(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Pie Chart":
            fig = px.pie(data_df, names=x_axis, values=y_axis)
        elif chart_type == "Histogram":
            fig = px.histogram(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Box Plot":
            fig = px.box(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Heatmap":
            fig = px.density_heatmap(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Violin Plot":
            fig = px.violin(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Area Chart":
            fig = px.area(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Funnel Chart":
            fig = px.funnel(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Word Cloud":
            # For Word Cloud, we only need text data from x_axis column
            text_data = data_df[x_axis].dropna().astype(str).tolist()
            text = ' '.join(text_data)
            
            excluded_words = {
            "check", "service", "rep", "km", "vehicle", "gaadi",
            "hai", "kar", "me", "ka", "ki", "ko", "se", "ke",
            "schedule", "washing", "1000", "10000", "maxicare",
            "wheel", "alignment", "balance", "pickup", "cleaning", "wash", "rahi", "nhi", "rha", "krne", "rhe", "hona", "par", "lag", "clean",
            "CLU", "ENG", "BOD", "CLN", "GEN", "STG", "WHT", "IFT", "BRK", "ELC", "TRN", "FUE", "HVA", "SER", "EPT", "SUS", "DRL", "EXH", "SAF", "VAS", "RE-",
            "708", "013", "405", "SWU"
            }

            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=200, stopwords=excluded_words).generate(text)
            
            # Convert to Plotly figure
            fig = px.imshow(wordcloud.to_array())
            fig.update_layout(
                
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
        return fig
    except Exception as e:
        logger.info(f"generate_chart_figure in main.py, Error generating {chart_type} chart: {str(e)}")
        raise
class ChartRequest(BaseModel):
    x_axis: str
    y_axis: str
    chart_type: str
    table_data: List[Dict]  # List of row dicts


@app.post("/generate-chart")
async def generate_chart(request0: ChartRequest):
    """
    Generates a chart based on the provided request data.
    Handles both numeric charts and text-based Word Cloud.
    """
    try:
        x_axis = request0.x_axis
        y_axis = request0.y_axis
        chart_type = request0.chart_type
        table_data = request0.table_data  # List of dicts

        # Convert list of dicts to DataFrame
        data_df = pd.DataFrame(table_data)

        # Validate columns exist
        if x_axis not in data_df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{x_axis}' not found in data")
        if chart_type != "Word Cloud" and y_axis not in data_df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{y_axis}' not found in data")

        # Data processing based on chart type
        if chart_type == "Word Cloud":
            if not pd.api.types.is_string_dtype(data_df[x_axis]):
                data_df[x_axis] = data_df[x_axis].astype(str)
        else:
            try:
                # data_df[y_axis] = pd.to_numeric(data_df[y_axis], errors='coerce')
                data_df = data_df.dropna(subset=[y_axis])
                if len(data_df) == 0:
                    raise ValueError("No valid numeric data available after conversion")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing numeric data: {str(e)}")

        # Generate the chart
        fig = generate_chart_figure(data_df, x_axis, y_axis, chart_type)
        if fig is None:
            raise HTTPException(status_code=400, detail="Unsupported chart type selected")

        return JSONResponse(content={"chart": fig.to_json()})

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
class TableDownloadRequest(BaseModel):
    table_name: str
    table_data: dict

@app.get("/download-table/")
@app.post("/download-table")
async def download_table(payload: TableDownloadRequest):
    """
    Downloads a table as an Excel file from sent JSON data.
    """
    table_name = payload.table_name
    data_dict = payload.table_data
    # Extract the list of rows from the dict
    rows = data_dict.get('Table data', [])
    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Generate Excel file (implement this function as you need)
    output = download_as_excel(df, filename=f"{table_name}.xlsx")

    # Return as streaming response
    response = StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response.headers["Content-Disposition"] = f"attachment; filename={table_name}.xlsx"
    return response
# Replace APIRouter with direct app.post
def format_number(x):
    # None check (in case of nulls)
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):d}"
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return f"{int(x):d}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.1f}"
    if isinstance(x, decimal.Decimal) and x == int(x):
        return f"{int(x):d}"
    if isinstance(x, decimal.Decimal):
        return f"{float(x):.1f}"
    return str(x)
@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file using Azure OpenAI's Whisper model.

    Args:
        file (UploadFile): The audio file to transcribe.

    Returns:
        JSONResponse: A JSON response containing the transcription or an error message.
    """
    try:
        # Check if API key is available
        if not AZURE_OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="Missing Azure OpenAI API Key.")
        
        # Read audio file
        audio_bytes = await file.read()
        audio_bio = BytesIO(audio_bytes)
        audio_bio.name = file.filename  # Use original filename or set appropriate extension

        # Transcribe using Azure OpenAI
        transcript = azure_openai_client.audio.transcriptions.create(
            model="whisper-1",  # Azure deployment name for Whisper model
            file=audio_bio
        )

        return {"transcription": transcript.text}

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error transcribing audio: {str(e)}"}, 
            status_code=500
        )

@app.get("/get_questions/")
@app.get("/get_questions")
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_questions(request: Request, subject: str):
    """
    Fetches questions from a CSV file in Azure Blob Storage based on the selected subject.

    Args:
        subject (str): The subject to fetch questions for.

    Returns:
        JSONResponse: A JSON response containing the list of questions or an error message.
    """
    question_type = request.session.get('current_question_type')
    if question_type == 'generic':
        csv_file_name = f"table_files/Azure-SQL-DB_questions_generic.csv"
    else: 
        csv_file_name = f"table_files/Azure-SQL-DB_questions.csv"
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=csv_file_name)

    try:
        # Check if the blob exists
        if not blob_client.exists():
            logger.error(f"file not found {csv_file_name}")
            return JSONResponse(
                content={"error": f"The file {csv_file_name} does not exist."}, status_code=404
            )

        # Download the blob content
        blob_content = blob_client.download_blob().content_as_text()

        # Read the CSV content
        questions_df = pd.read_csv(StringIO(blob_content))
        
        if "question" in questions_df.columns:
            questions = questions_df["question"].tolist()
        else:
            questions = questions_df.iloc[:, 0].tolist()

        return {"questions": questions}

    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while reading the file: {str(e)}"}, status_code=500
        )
# Function to load prompts from YAML

def load_prompts(filename:str):
    """
    Loads prompts from the chatbot_prompt.yaml file.

    Returns:
        dict: A dictionary containing the loaded prompts.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.info(f"load_prompts in main.py: Error reading prompts file: {e}")
        return {}
    

# @app.post("/submit_feedback/")
# @app.post("/submit_feedback")
# async def submit_feedback(request: Request):
#     data = await request.json() # Corrected for FastAPI
    
#     table_name = data.get("table_name")
#     feedback_type = data.get("feedback_type")
#     user_query = data.get("user_query")
#     sql_query = data.get("sql_query")

#     if not table_name or not feedback_type:
#         return JSONResponse(content={"success": False, "message": "Table name and feedback type are required."}, status_code=400)

#     try:
#         # Create database connection
#         engine = create_engine(
#         f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}'
#         )
#         Session = sessionmaker(bind=engine)
#         session = Session()

#         # Sanitize input (Escape single quotes)
#         table_name = escape_single_quotes(table_name)
#         user_query = escape_single_quotes(user_query)
#         sql_query = escape_single_quotes(sql_query)
#         feedback_type = escape_single_quotes(feedback_type)

#         # Insert feedback into database
#         insert_query = f"""
#         INSERT INTO lz_feedbacks (department, user_query, sql_query, table_name, data, feedback_type, feedback)
#         VALUES ('unknown', :user_query, :sql_query, :table_name, 'no data', :feedback_type, 'user feedback')
#         """

#         session.execute(insert_query, {
#         "table_name": table_name,
#         "user_query": user_query,
#         "sql_query": sql_query,
#         "feedback_type": feedback_type
#         })

#         session.commit()
#         session.close()

#         return JSONResponse(content={"success": True, "message": "Feedback submitted successfully!"})

#     except Exception as e:
#         session.rollback()
#         session.close()
#         return JSONResponse(content={"success": False, "message": f"Error submitting feedback: {str(e)}"}, status_code=500)


import csv

def get_keyphrases():
    keyphrases = []
    with open('table_files/keyphrases_rephrasing.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assumes the column is named exactly 'keyphrases'
            if 'keyphrases' in row and row['keyphrases']:
                keyphrases.append(row['keyphrases'])
    return ','.join(keyphrases)

# if 'messages' not in session_state:
#     session_state['messages'] = []
    
def parse_table_data(csv_file_path):
    """
    Parses a CSV file containing table definitions and returns structured data.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        dict: A dictionary with table names as keys and their metadata as values
              Format: {
                  'table_name': {
                      'description': 'table description',
                      'columns': [
                          {
                              'name': 'column_name',
                              'type': 'data_type',
                              'nullable': boolean,
                              'description': 'column description'
                          },
                          ...
                      ]
                  },
                  ...
              }
    """
    tables = defaultdict(lambda: {
        'description': '',
        'columns': []
    })
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if len(row) < 3:  # Skip incomplete rows
                continue
                
            table_name = row[0].strip()
            table_description = row[1].strip()
            column_info = row[2].strip()
            
            # Parse column information (name, type, nullable, description)
            if '(' in column_info:
                # Extract column name and type
                col_name = column_info.split('(')[0].strip()
                type_part = column_info.split('(')[1].split(')')[0].strip()
                
                # Check for NULLABLE
                nullable = 'NULLABLE' in column_info
                
                # Extract description (after colon if present)
                if ':' in column_info:
                    col_desc = column_info.split(':')[-1].strip()
                else:
                    col_desc = ''
            else:
                col_name = column_info
                type_part = ''
                nullable = False
                col_desc = ''
            
            # Ensure table exists in dictionary
            if table_name not in tables:
                tables[table_name]['description'] = table_description
            
            # Add column information
            tables[table_name]['columns'].append({
                'name': col_name,
                'type': type_part,
                'nullable': nullable,
                'description': col_desc
            })
    
    return dict(tables)


@app.post("/submit")
@limiter.limit(SENSITIVE_RATE_LIMIT) 
async def submit_query(
    request: Request,
    section: str = Form(...),
    database: str = Form(...), 
    user_query: str = Form(...),
    page: int = Query(1),
    records_per_page: int = Query(10),
    model: Optional[str] = Form(AZURE_DEPLOYMENT_NAME),
    db: Session = Depends(get_db),

):
    logger.info(f"Endpoint:  /submit request with query: {user_query}, section: {section}, database: {database}")
    with log_execution_time("Submit Endpoint in main"):
        try:
        # Initialize response structure
            response_data = {
                "user_query": user_query,
                "query": "",
                "tables": [],
                "llm_response": "",
                "chat_response": "",
                "history":  request.session.get('messages', []),
                "interprompt": "",
                "langprompt": "",
                "error": None
            }
            # with log_execution_time("Prompt loading and rephrase LLM"):
            try:
                # Reset per-request variables
                unified_prompt = ""
                final_prompt = ""
                llm_reframed_query = ""

                # Get current question type from session
                current_question_type = request.session.get("current_question_type", "generic")
                # prompts = request.session.get("prompts", load_prompts("generic_prompt.yaml")
                prompts = load_prompts("generic_prompt.yaml")
                request.session['user_query'] = user_query  # Still store original query separately if needed

                # Handle session messages
                if "messages" not in request.session:
                    request.session["messages"] = []
                
                # Don't add user_query to messages yet - we'll add the reframed version later
                chat_history = ""
                if request.session['messages']:  # Check if messages exist (should contain at most 1)
                    last_msg = request.session['messages'][-1]  # Get the only message
                    chat_history = f"{last_msg['role']}: {last_msg['content']}"
                
                logger.info(f"Inside /submit request, Chat history: {chat_history}")
                # logger.info(f"Messages in session for new question: {request.session['messages']}")
                # Step 1: Generate unified prompt based on question type
                try:
                    logger.info(f"Inside /submit request and user has chosen  {current_question_type}.")

                    if current_question_type == "usecase":
                        key_parameters = get_key_parameters()
                        keyphrases = get_keyphrases()
                        unified_prompt = prompts["unified_prompt"].format(
                            user_query=user_query,
                            chat_history=chat_history,
                            key_parameters=key_parameters,
                            keyphrases=keyphrases
                        )
                        
                        # llm_reframed_query = llm.invoke(unified_prompt).content.strip()
                        with log_execution_time("submit_query -> Rephrasing LLM"):
                            response = azure_openai_client.chat.completions.create(
                                model=AZURE_DEPLOYMENT_NAME,
                                store= True,
                                
                                messages=[
                                {"role": "system", "content": unified_prompt},
                                {"role": "user", "content": user_query}
                            ],
                            temperature=0,  # Lower temperature for more predictable, structured output
                            response_format={"type": "json_object"}  # This is the key parameter!
                            )
                    # The response content will be a JSON string
                        response_content = response.choices[0].message.content
                        logger.info(f"Inside submit function: rephrased response form LLM is:: {response_content}")

                        # Parse the guaranteed JSON string into a Python dictionary
                        json_output = json.loads(response_content)
                        # logger.info(f"Inside submit function, usecase: json output in usecase: {json_output}")
                        # Now you can safely access the keys
                        llm_reframed_query = json_output.get("rephrased_query")

                        intent_result = intent_classification(llm_reframed_query)
                        
                        if not intent_result:
                            error_msg = "Please rephrase or add more details to your question as I am not able to assess the Intended Use case"
                            
                            
                            response_data = {
                                "user_query": user_query,
                                "query": "",
                                "tables": "",
                                "llm_response": llm_reframed_query,
                                "chat_response": error_msg,
                                "history": request.session['messages'],
                                "interprompt": unified_prompt,
                                "langprompt": ""
                            }
                            return JSONResponse(content=response_data)
                        chosen_tables = db_tables
                        selected_business_rule = get_business_rule(intent_result["intent"])
                        examples = get_examples(llm_reframed_query, "usecase", intent = intent_result["intent"])
                    elif current_question_type == "generic":
                        tables_metadata = get_table_metadata()
                        unified_prompt = prompts["unified_prompt"].format(
                            user_query=user_query,
                            chat_history=chat_history,
                            key_parameters=get_key_parameters(),
                            keyphrases=get_keyphrases(),
                            table_metadata=tables_metadata
                        )
                        
                        # llm_response_str = llm.invoke(unified_prompt).content.strip()
                        with log_execution_time("submit_query -> rephrase LLM"):

                            response = azure_openai_client.chat.completions.create(
                                model=AZURE_DEPLOYMENT_NAME,
                                store = True,
                                messages=[
                                {"role": "system", "content": unified_prompt},
                                {"role": "user", "content": user_query}
                            ],
                            temperature=0,  # Lower temperature for more predictable, structured output
                            response_format={"type": "json_object"}  # This is the key parameter!
                            )

                    # The response content will be a JSON string
                        response_content = response.choices[0].message.content
                        logger.info(f"Inside submit function, generic: rephrased query from LLM recieved is: {response_content}")

                        # Parse the guaranteed JSON string into a Python dictionary
                        json_output = json.loads(response_content)

                        # Now you can safely access the keys
                        # llm_reframed_query = json_output.get("rephrased_query")
                        try:
                            # llm_result = json.loads(llm_response_str)
                            llm_reframed_query = json_output.get("rephrased_query", "")
                            chosen_tables = db_tables
                            selected_business_rule = ""
                            logger.info(f"Inside submit function, generic: chosen tables are: {chosen_tables}")
                            with log_execution_time("submit_query -> examples"):
                                examples = get_examples(llm_reframed_query, "generic")

                        except json.JSONDecodeError:
                            raise HTTPException(
                                status_code=500,
                                detail="Failed to parse LLM response"
                            )
                    
                    # Now add the reframed query to messages instead of original user_query
                    # logger.info(f"Now, adding message to history: {llm_reframed_query}")
                    request.session['messages'] = [{"role": "user", "content": llm_reframed_query}]
                    # logger.info(f"messages in session: {request.session['messages']}")
                    response_data["llm_response"] = llm_reframed_query
                    response_data["interprompt"] = unified_prompt
                    
                except Exception as e:
                    logger.error(f"Prompt generation error: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Prompt generation failed: {str(e)}"
                    )

                # Rest of your code remains the same...
                # Step 2: Invoke LangChain
                try:
                    relationships = find_relationships_for_tables(chosen_tables , 'table_relation.yaml')
                    table_details = get_table_details(table_name=chosen_tables)
                    # logger.info(f"Inside /submit request, relationships: {relationships}")
                    # logger.info(f"messages in session just before invoke chain: {request.session['messages']}")
                    print("final",final_query_instruction )

                    response, chosen_tables, tables_data, final_prompt, description= invoke_chain(
                        db,
                        llm_reframed_query,  # Using the reframed query here
                        request.session['messages'],
                        model,
                        section,
                        database,
                        table_details,
                        selected_business_rule,
                        current_question_type,
                        relationships,
                        examples,
                        final_query_instruction

                    )
                    
                    response_data["langprompt"] = str(final_prompt)
                    response_data["description"] = description

                    
                    if isinstance(response, str):
                        request.session['generated_query'] = response
                        response_data["query"] = response
                        request.session['generated_query'] = response
                    else:
                        response_data["query"] = response.get("query", "")
                        request.session['generated_query'] = response.get("query", "")
                        request.session['chosen_tables'] = chosen_tables
                        # request.session['tables_data'] = tables_data

                except Exception as e:
                    logger.error(f"LangChain invocation error: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Query execution failed: {str(e)}"
                    )

                # Step 3: Process results
                try:
                    # Format numeric columns
                    for table_name, df in tables_data.items():
                        for col in df.select_dtypes(include=['number']).columns:
                            tables_data["Table data"][col] = df[col].apply(format_number)
                    
                    tables_data_dict = {k: v.to_dict(orient='records') for k, v in tables_data.items()}

                    # Prepare table HTML
                    initial_page_html = prepare_table_html(tables_data, 1,10)

                    response_data["tables"] = initial_page_html
                    response_data["tables_data"] = tables_data_dict           
                # Generate insights if data exists
                    # data_preview = next(iter(session_state['tables_data'].values())).head(5).to_string(index=False)
                    # response_data["chat_response"] = ""  # Placeholder for actual insights
                    
                except Exception as e:
                    logger.error(f"Data processing error: {str(e)}")
                    response_data["chat_response"] = f"Data retrieved but processing failed: {str(e)}"

                # Append successful response to chat history
                # session_state['messages'].append({
                #     "role": "assistant",
                #     "content": response_data["chat_response"]
                # })

                return JSONResponse(content=convert_dates(response_data))

            except HTTPException as he:
                # Capture error details
                response_data.update({
                    "chat_response": f"Error: {he.detail}",
                    "error": str(he.detail),
                    "history": request.session['messages'],
                    "langprompt": str(final_prompt) if 'final_prompt' in locals() else "Not generated due to error",
                    "interprompt": unified_prompt if 'unified_prompt' in locals() else "Not generated due to error"
                })
                
                
                return JSONResponse(
                    content=response_data,
                    status_code=he.status_code
                )
                
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                response_data.update({
                    "chat_response": "An unexpected error occurred",
                    "error": str(e),
                    "history":  request.session['messages'],
                    "langprompt": str(final_prompt) if 'final_prompt' in locals() else "Not generated due to error",
                    "interprompt": unified_prompt if 'unified_prompt' in locals() else "Not generated due to error"
                })
                
                request.session['messages'].append({
                    "role": "user",
                    "content": "An unexpected error occurred"
                })
                
                return JSONResponse(
                    content=response_data,
                    status_code=500
                )
        except Exception as e:
            logger.error(f"Error in Submit Endpoint: {e}")
# Replace APIRouter with direct app.post

@app.post("/reset-session")
@limiter.limit(DEFAULT_RATE_LIMIT)
async def reset_session(request: Request):
    """
    Resets the session state by clearing the session dictionary.
    """
    request.session.clear()  # Clear all session data

    # Set default session variables
    request.session['messages'] = []
    request.session["current_question_type"] = "generic"
    # request.session["prompts"] = load_prompts("generic_prompt.yaml")

    logger.info(f"Endpoint: reset sesion, Question type is: {request.session.get('current_question_type')}")
    return {"message": "Session state cleared successfully"}

def prepare_table_html(tables_data, page_number, records_per_page):
    """
    Prepares HTML for displaying table data with pagination (client-side version).
    Returns the first page by default.
    """
    tables_html = []
    for table_name, data in tables_data.items():
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page
        
        # Get the first page by default
        start_index = (page_number - 1) * records_per_page
        end_index = start_index + records_per_page
        page_data = data.iloc[start_index:end_index]
        
        # Generate styled HTML
        styled_html = display_table_with_styles(page_data, table_name)
        
        tables_html.append({
            "table_name": table_name,
            "table_html": styled_html,
            "pagination": {
                "current_page": page_number,
                "total_pages": total_pages,
                "records_per_page": records_per_page,
                "total_records": total_records
            }
        })
    return tables_html
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the root HTML page.

    Args:
        request (Request): The incoming request.

    Returns:
        TemplateResponse: The rendered HTML template.
    """
    logger.info("Endpoint: /, main read route ")
    with log_execution_time("read-route in main"):
        try:
            # Extract table names dynamically
            request.session.clear()
            tables = []
            # Only set defaults if not already set
            if "current_question_type" not in request.session:
                request.session["current_question_type"] = "generic"
                # request.session["prompts"] = load_prompts("generic_prompt.yaml")

            # Pass dynamically populated dropdown options to the template
            return templates.TemplateResponse("index.html", {
                "request": request,
                "databases": databases,                                     
                "tables": tables,        # Table dropdown based on database selection
                # "question_dropdown": question_dropdown.split(','),  # Static questions from env
            })
        except Exception as e:
            logger.error(f"Error in main read_root: {e}")

# Table data display endpoint
def display_table_with_styles(data, table_name):
    """
    Displays a Pandas DataFrame as an HTML table with custom styles.
    """
    # Ensure that the index starts from 1
    data.index = range(1, len(data) + 1)
    
    styled_table = (
        data.style
        .set_table_attributes('class="data-table" style="border: 2px solid black; border-collapse: collapse;"')
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#333'), 
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('font-size', '16px')
            ]},
            {'selector': 'td', 'props': [
                ('border', '2px solid black'),
                ('padding', '5px')
            ]}
        ])
        .to_html(escape=False)
    )
    return styled_table

# @app.get("/get_table_data/")
# @app.get("/get_table_data")
# async def get_table_data(
#     request:Request,

#     table_name: str = Query(...),
#     page_number: int = Query(1),
#     records_per_page: int = Query(10),
# ):
#     """Fetch paginated and styled table data."""
#     try:
#         # Check if the requested table exists in the tables_data from the initial response
#         if "tables_data" not in request.query_params or table_name not in request.query_params["tables_data"]:
#             raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

#         # Retrieve the data for the specified table from the query params
#         data = request.query_params["tables_data"][table_name]
#         total_records = len(data)
#         total_pages = (total_records + records_per_page - 1) // records_per_page

#         # Ensure valid page number
#         if page_number < 1 or page_number > total_pages:
#             raise HTTPException(status_code=400, detail="Invalid page number.")

#         # Slice data for the requested page
#         start_index = (page_number - 1) * records_per_page
#         end_index = start_index + records_per_page
#         page_data = data.iloc[start_index:end_index]

#         # Style the table as HTML
#         styled_table = (
#             page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"')
#             .set_table_styles([
#                 {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]},
#                 {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
#             ])
#             .to_html(escape=False)  # Render as HTML
#         )

#         return {
#             "table_html": styled_table,
#             "page_number": page_number,
#             "total_pages": total_pages,
#             "total_records": total_records,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating table data: {str(e)}")
class QuestionTypeRequest(BaseModel):
    question_type: str
@app.post("/set-question-type")
@limiter.limit(DEFAULT_RATE_LIMIT)
async def set_question_type(payload: QuestionTypeRequest, request: Request):
    current_question_type = payload.question_type
    filename = "generic_prompt.yaml" if current_question_type == "generic" else "chatbot_prompt.yaml"
    prompts = load_prompts(filename)
    request.session["current_question_type"] = current_question_type
    # request.session["prompts"] = prompts  # If you want to store prompts per session

    return JSONResponse(content={"message": "Question type set", "prompts": prompts})

