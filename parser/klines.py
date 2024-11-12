from typing import List
from datetime import datetime
import pendulum

import asyncio

from binance.client import AsyncClient
from .config import BinanceConfig as Config

from sqlalchemy.future import select
from sqlalchemy import text
from database.Base import async_connection
from database.models.token import Symbol, Kline



async def get_symbols() -> List[Symbol]:
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


async def save_new_klines(symbol: Symbol, klines: List[List], max_time_open: float = 0.0) -> None:
    """Insert new klines into the database.

    Args:
        symbol (Symbol): ORM Symbol object.
        klines (List[List]): Raw kline data.
        max_time_open (float, optional): Last `time_open` property value in `symbol` table. Defaults to 0.0.
    """
    session, engine = async_connection()
    try:
        async with session() as db:
            new_klines = filter(lambda kline: pendulum.from_timestamp(kline[0]/1000, "UTC").naive() > max_time_open, klines)
            klines_for_db = [
                Kline(
                    id_symbol=symbol.id,
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    number_of_trades=kline[8],
                    time_open=pendulum.from_timestamp(kline[0]/1000, "UTC").naive(),
                    time_close=pendulum.from_timestamp(kline[6]/1000, "UTC").naive()
                ) for kline in new_klines if datetime.now() >= datetime.fromtimestamp(kline[6]/1000)
            ]
            db.add_all(klines_for_db)
            await db.commit()
    finally:
        await engine.dispose()


async def main():
    """Parse new klines from Binance API and update the database."""
    # Init Binance API object.
    BinanceAPI = AsyncClient(Config.api_key, Config.api_secret)
    # Init database session.
    session, engine = async_connection()

    try:
        # Fetch symbol IDs and last kline time_opens according to the symbols.
        sql_query = text(f"""select id_symbol, max(time_open) from kline group by id_symbol""")
        async with session() as db:
            result = await db.execute(sql_query)
            max_time_open_in_db = {element[0]: element[1].timestamp() * 1000 if element[1] else 0 for element in result.all()}

        # Fetch symbols and tickers from DB.
        symbols = await get_symbols()

        # Process symbols.
        for symbol in symbols:
            # Get last 10 candles in '1m' time frame from the API.
            klines = await BinanceAPI.get_klines(symbol=symbol.symbol, interval="1m", limit=10)
            # Extract last kline time_open from the DB.
            max_time_open = datetime.fromtimestamp(max_time_open_in_db[symbol.id]/1000 if symbol.id in max_time_open_in_db else 0.0)
            # Save new klines.
            await save_new_klines(symbol, klines, max_time_open)
            
    except Exception as e:
        print(f"An error occurred: {e}")    
    finally:
        await BinanceAPI.close_connection()
        await engine.dispose()
    

if __name__ == "__main__":
    asyncio.run(main())
