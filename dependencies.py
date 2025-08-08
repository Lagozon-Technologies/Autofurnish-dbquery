from fastapi import Request

def get_llm(request: Request):
    return request.app.state.azure_openai_client

def get_embeddings(request: Request):
    return request.app.state.schema_collection

def get_db(request: Request):
    SessionLocal = request.app.state.SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# def get_db(request: Request):
#     SessionLocal = request.app.state.SessionLocal
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def get_redis(request: Request):
#     return request.app.state.redis_client
