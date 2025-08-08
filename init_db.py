import asyncio
from database import engine
from models import metadata

async def main():
    try:
        metadata.create_all(bind=engine)
        print("Tables created successfully.")
        print("Tables available: ", metadata.tables.keys())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
