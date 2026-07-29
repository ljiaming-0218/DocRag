from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from stores.user_store import create_user_indexes
from stores.database import close_database, connect_database
from stores.message_store import create_message_indexes
from stores.conversation_store import create_conversation_indexes
from stores.document_store import create_document_indexes

from config import FRONTEND_DIR

from routers.user_router import router as user_router
from routers.conversation_router import router as conversation_router
from routers.pdf_router import router as pdf_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await connect_database()
        await create_conversation_indexes()
        await create_message_indexes()
        await create_user_indexes()
        await create_document_indexes()
        yield
    finally:
        await close_database()


app = FastAPI(
    title="DocRAG Agent PDF 文献阅读助手",
    lifespan=lifespan,
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pdf_router)
app.include_router(conversation_router)
app.include_router(user_router)
@app.get("/health")
def health_check() -> dict:
    return {"提示": "DocRAG 后端服务正在运行"}


app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)
