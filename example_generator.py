import json
import datetime
import os
import pyodbc
from dotenv import load_dotenv
from decimal import Decimal

# Load environment variables
load_dotenv()

# Read Azure SQL credentials from .env
AZURE_SQL_SERVER = os.getenv("SQL_DB_SERVER")
AZURE_SQL_PORT = os.getenv("SQL_DB_PORT", "1433")
AZURE_SQL_DATABASE = os.getenv("SQL_DB_NAME")
AZURE_SQL_USERNAME = os.getenv("SQL_DB_USER")
AZURE_SQL_PASSWORD = os.getenv("SQL_DB_PASSWORD")
AZURE_SQL_DRIVER = os.getenv("SQL_DB_DRIVER", "ODBC Driver 18 for SQL Server")

# Connection string
conn_str = (
    f"DRIVER={{{AZURE_SQL_DRIVER}}};"
    f"SERVER={AZURE_SQL_SERVER},{AZURE_SQL_PORT};"
    f"DATABASE={AZURE_SQL_DATABASE};"
    f"UID={AZURE_SQL_USERNAME};"
    f"PWD={AZURE_SQL_PASSWORD};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=no;"
    f"Connection Timeout=30;"
)

# Establish connection
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Load metadata JSON
with open("table_files\\expanded_columns.json", "r") as f:
    metadata = json.load(f)

# Custom JSON encoder to handle Decimal and other non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

# Helper to format numbers
def format_number(x):
    if isinstance(x, int):
        return f"{x:d}"
    elif isinstance(x, float) and x.is_integer():
        return f"{int(x):d}"
    else:
        return f"{x:.1f}"

# Regenerate examples
def regenerate_examples(column_info, limit=5):
    try:
        table = column_info["metadata"]["table_name"]
        column = column_info["column_name"].split(".")[-1]
        data_type = column_info["metadata"].get("data_type", "").upper()

        query = f"""
        SELECT DISTINCT TOP {limit} [{column}]
        FROM [{table}]
        WHERE [{column}] IS NOT NULL
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        values = []

        for row in rows:
            value = row[0]
            if value is None:
                continue

            # Convert Decimal to float for JSON serialization
            if isinstance(value, Decimal):
                value = float(value)
            elif data_type == "INTEGER":
                value = int(value)
            elif data_type == "FLOAT":
                value = float(value)
            elif data_type == "STRING":
                value = str(value)
            elif data_type in ("DATE", "DATETIME", "TIMESTAMP") and isinstance(value, (datetime.date, datetime.datetime)):
                value = value.isoformat()

            values.append(value)

        column_info["examples"] = values

    except Exception as e:
        print(f"❌ Error processing {column_info.get('column_name')}: {e}")
        column_info["examples"] = []

# Regenerate examples for each column in metadata
for col in metadata:
    regenerate_examples(col)

# Save updated metadata with custom encoder
with open("metadata_updated.json", "w") as f:
    json.dump(metadata, f, indent=2, cls=CustomJSONEncoder)

print("✅ metadata_updated.json generated with refreshed examples from Azure SQL.")

# Close connection
cursor.close()
conn.close()