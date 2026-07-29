

from stores.database import get_database


async def insert_user(user: dict) -> None:
    database = get_database()
    user_collection = database["users"]
    await user_collection.insert_one(user)


async def find_user_by_username(username: str) -> dict | None:
    database = get_database()
    user_collection = database["users"]
    return await user_collection.find_one({"username": username})


async def find_user_by_id(user_id: str) -> dict | None:
    database = get_database()
    user_collection = database["users"]
    return await user_collection.find_one({"_id": user_id})


async def create_user_indexes() -> None:
    database = get_database()
    user_collection = database["users"]
    await user_collection.create_index(
        [("username", 1)],
        name="username_unique_idx",
        unique=True
    )
