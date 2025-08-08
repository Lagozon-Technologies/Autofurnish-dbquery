import pandas as pd
import os, json
from operator import itemgetter
# from langchain.chains.openai_tools import create_extraction_chain_pydantic 
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI 
from openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI
import platform


AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME')

# llm = AzureChatOpenAI(
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     azure_deployment=AZURE_DEPLOYMENT_NAME,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     temperature=0
# )

from typing import List

os_name = platform.system()
if os_name == "Windows":
    # Do something for Windows
    print("Running on Windows")
elif os_name == "Linux":
    # Do something for Linux
    print("Running on Linux")
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
else:
    # Raise an exception for unsupported OS
    raise RuntimeError(f"Unsupported OS: {os_name}")
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
def get_table_details(table_name=None):
    """
    Returns details for one or more tables from hardcoded JSON files.
    - table_name: string or list of strings (table names to filter on)
    """
    # Hardcoded file paths
    table_path = os.path.join('table_files', 'expanded_tables.json')
    column_path = os.path.join('table_files', 'expanded_columns.json')
    
    # Load table data
    try:
        with open(table_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
    except FileNotFoundError:
        return f"File not found: {table_path}"
    except Exception as e:
        return f"Error reading table file: {e}"

    if not isinstance(table_data, list):
        return "Invalid table JSON format: must be a list of table objects."
    
    # Load column data
    try:
        with open(column_path, 'r', encoding='utf-8') as f:
            column_data = json.load(f)
    except FileNotFoundError:
        return f"File not found: {column_path}"
    except Exception as e:
        return f"Error reading column file: {e}"

    if not isinstance(column_data, list):
        return "Invalid column JSON format: must be a list of column objects."
    
    # Normalize table_name(s) for filtering
    table_names = []
    if table_name:
        if isinstance(table_name, str):
            table_names = [t.strip().lower() for t in table_name.split(';') if t.strip()]
        elif isinstance(table_name, list):
            table_names = [t.strip().lower() for t in table_name if t.strip()]
        else:
            return "Invalid table_name argument."
        filtered_tables = [t for t in table_data if t.get('id', '').strip().lower() in table_names]
        if not filtered_tables:
            return f"No details found for table(s): {', '.join(table_names)}"
    else:
        filtered_tables = table_data

    # Build details for each table
    table_details = ""
    for table in filtered_tables:
        tname = table.get('id', '').strip()
        tdesc = table.get('document', '').strip()  # or 'table_description' if that's your key
        table_details += f"Table Name: {tname}\n"
        table_details += f"Table Description: {tdesc}\n"
        table_details += "Columns:\n"
        
        # Find columns for this table
        columns = [col for col in column_data if col.get('metadata', {}).get('table_name', '').strip().lower() == tname.lower()]
        if columns:
            for col in columns:
                col_meta = col.get('metadata', {})
                col_name = col.get('column_name', '').split('.')[-1]
                data_type = col_meta.get('data_type', '')
                nullable = col_meta.get('nullable', False)
                description = col.get('column_desc', '')
                is_pk = col_meta.get('is_primary_key', False)
                is_fk = col_meta.get('is_foreign_key', False)
                flags = []
                if is_pk: flags.append("PK")
                if is_fk: flags.append("FK")
                if nullable: flags.append("NULLABLE")
                flag_str = " [" + ", ".join(flags) + "]" if flags else ""
                examples = col.get('examples', None)
                example_str = f" Example: {examples}" if examples is not None else "no example for this column"
                table_details += f"  - {col_name} ({data_type}){flag_str}: {example_str} {description} \n"
        else:
            table_details += "  No column details found.\n"
        table_details += "\n"

    if not table_details.strip():
        table_details = "No tables found in the JSON."
    
    return table_details.strip()
class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables

def get_table_metadata(json_filename='expanded_tables.json'):
    """
    Returns a list of table names and their descriptions from a JSON array file.
    - json_filename: name of the JSON file (should be an array of objects)
    """
    path = f'table_files/{json_filename}'
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    if not isinstance(data, list):
        return "JSON file must be a list of table objects."

    table_info = ""
    seen = set()
    for table in data:
        table_name = table.get('id')
        table_description = table.get('document')
        if not table_name or not table_description:
            continue
        table_name = table_name.strip()
        if table_name not in seen:
            seen.add(table_name)
            table_info += f"Table Name: {table_name}\nDescription: {table_description}\n\n"

    return table_info.strip()
# table_names = "\n".join(db.get_usable_table_names())
# table_details = get_table_details()
# print("testinf details",table_details, type(table_details))
# table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
#     The permissible tables names are listed below and must be strictly followed:

#     {table_details}

#     Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
# table_details_set_prompt = os.getenv('TABLE_DETAILS_SET_PROMPT')
# table_details_prompt=table_details_set_prompt.format(table=table_details)
# # print("Table_details_prompt: ", table_details_prompt)
# table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables
# mock_question_test = "How many product view by products in last week"
# table_chain_check = table_chain.invoke({"question":mock_question_test})
# print("test table chain  first mock_question  :" , mock_question_test ,"  Now tables selected:... ",table_chain_check)
