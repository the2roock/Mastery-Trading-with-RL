from typing import List
from datetime import datetime
import pytz
import asyncio

from binance.client import AsyncClient
from .config import BinanceConfig as Config

from sqlalchemy.future import select
from database.Base import async_connection
from database.models.token import Symbol, OrderBook



async def get_symbols() -> list[Symbol]:
    """Fetch symbol records from the database.

    Returns:
        List[Symbol]: List of ORM Symbol objects.
    """
    session, engine = async_connection()
    try:
        async with session() as db:
            result = await db.execute(select(Symbol).filter(Symbol.status == 1))
            symbols = [element[0] for element in result.all()]
    finally:
        await engine.dispose()
    return symbols


async def save_depths(symbol: Symbol, order_book: List[List]) -> None:
    """Insert new depths into the database.

    Args:
        symbol (Symbol): ORM Symbol object.
        order_book (List[List]): Raw depth data.
    """
    session, engine = async_connection()
    try:
        async with session() as db:
            depth_for_db = OrderBook(
                    id_symbol=symbol.id,
                    ask=order_book["ask"],
                    bid=order_book["bid"],
                    time_open=datetime.now(pytz.utc).replace(second=0, microsecond=0, tzinfo=None)
                )
            db.add(depth_for_db)
            await db.commit()
    finally:
        await engine.dispose()


async def main():
    """Parse new order book values from Binance API and update the database."""
    # Init Binance API object.
    BinanceAPI = AsyncClient(Config.api_key, Config.api_secret)
    
    try:
        # Fetch symbols and tickers from DB.
        symbols = await get_symbols()
        
        # Process symbols.
        for symbol in symbols:
            # Get current order book from the API.
            depth = await BinanceAPI.get_order_book(symbol=symbol.symbol, limit=1000)
            # Change depth`s data format to text.
            order_book = {
                "bid": " ".join([f"{float(element[0])}:{float(element[1])}" for element in depth["bids"][::-1]]),
                "ask": " ".join([f"{float(element[0])}:{float(element[1])}" for element in depth["asks"]]),
            }
            # Save new order book.
            await save_depths(symbol, order_book)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await BinanceAPI.close_connection()
    

if __name__ == "__main__":
    asyncio.run(main())
