from typing import List
import asyncio

from binance.client import AsyncClient
from .config import BinanceConfig as Config

from sqlalchemy.future import select
from database.Base import async_connection
from database.models.token import Symbol


async def fetch_symbols() -> List[str]:
    """Fetch symbol records from the database.

    Returns:
        List[str]: List of symbols` tikers.
    """
    session, engine = async_connection()
    try:
        async with session() as db:
            symbols = await db.execute(select(Symbol))
            all_symbols = [s[0].symbol for s in symbols.all()]
    finally:
        await engine.dispose()
    return all_symbols


async def save_new_symbols(new_symbols: List[str]) -> None:
    """Insert new symbols into the database.

    Args:
        new_symbols (List[str]): List of new symbols` tickers.
    """
    symbols = [
        Symbol(
            symbol=symbol,
            data={
                "src": "binance-api"
            }
        ) for symbol in new_symbols
    ]
    session, engine = async_connection()
    try:
        async with session() as db:
            db.add_all(symbols)
            await db.commit()
    finally:
        await engine.dispose()


async def main():
    """Parse new tickers from Binance API and update the database."""
    # Init Binance API object.
    BinanceAPI = AsyncClient(Config.api_key, Config.api_secret)
    
    # Process parsing the API and updating the DB.
    try:
        # Fetch symbols from the DB.
        symbols_from_db = await fetch_symbols()
        
        # Fetch symbols from Binance.
        symbols_from_binance = await BinanceAPI.get_all_tickers()
        symbols_from_binance = [symbol["symbol"] for symbol in symbols_from_binance]
        
        # Select and savenew symbols.
        new_symbols = list(set(symbols_from_binance) - set(symbols_from_db))
        if new_symbols:
            await save_new_symbols(new_symbols)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await BinanceAPI.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
