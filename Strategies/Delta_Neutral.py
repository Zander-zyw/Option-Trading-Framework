### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
### import DeribitClient ###

from Deribit.DeribitClient import DeribitClient
import asyncio

async def main():
    client = DeribitClient()
    connect_task = asyncio.create_task(client.connect())
    await asyncio.sleep(3)

    await client.get_order_book("BTC-PERPETUAL", 1)

    await client.disconnect()
    await connect_task

if __name__ == "__main__":
    asyncio.run(main())