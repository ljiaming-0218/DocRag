from pymongo import AsyncMongoClient

from config import MONGODB_DB_NAME, MONGODB_URI


if not MONGODB_URI or not MONGODB_DB_NAME:
    raise RuntimeError("MongoDB 配置不完整")

client = None
database = None

async def connect_database():
    
    global client, database
    client = AsyncMongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
    )
    await client.admin.command({"ping": 1})
    database = client[MONGODB_DB_NAME]

def get_database():
    if database is None:
        raise RuntimeError("MongoDB 尚未连接")
    return database


async def ping_database() -> bool:
    if client is None:
        return False

    await client.admin.command({"ping": 1})
    return True


async def close_database():
    global client, database

    if client is not None:
        await client.close()

    client = None
    database = None
