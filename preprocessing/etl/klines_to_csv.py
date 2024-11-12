import sys
import os
from typing import List

import pandas as pd

from sqlalchemy.orm import Session

from database.Base import connection
from database.models.token import Symbol, Kline


def fetch_symbol(db: Session, symbol_name: str) -> Symbol:
    """Fetch symbol record from the database.

    Args:
        db (Session): Database session.
        symbol_name (str): Target symbol.

    Returns:
        Symbol: ORM Symbol object or None if not found.
    """
    return db.query(Symbol).filter(Symbol.symbol == symbol_name).one_or_none()


def fetch_klines(db: Session, symbol_id: int) -> List[Kline]:
    """Fetch klines associated with the given symbol ID.

    Args:
        db (Session): Database session.
        symbol_id (int): Target symbol`s ID.

    Returns:
        List[Kline]: List of ORM Kline objects.
    """
    return db.query(Kline).filter(Kline.id_symbol == symbol_id).all()


def process_klines(klines: List) -> pd.DataFrame:
    """Convert klines to DataFrame and process it.

    Args:
        klines (List): Raw kline data.

    Returns:
        pd.DataFrame: Processed kline DataFrame.
    """
    df = pd.DataFrame([kline.__dict__ for kline in klines])
    df.drop(columns=["_sa_instance_state"], inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=["id", "id_symbol", "time_create", "time_update", "time_close"], inplace=True)
    
    # Rename columns
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    
    # Convert time_open to datetime and set it as index
    df['time_open'] = pd.to_datetime(df['time_open'])
    df.set_index('time_open', inplace=True)
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and check the DataFrame.

    Args:
        df (pd.DataFrame): Kline DataFrame object.

    Returns:
        pd.DataFrame: Cleaned kline DataFrame.
    """
    # Remove duplicates
    duplicates = len(df) - len(df.drop_duplicates())
    df.drop_duplicates(inplace=True)
    index_duplicates = sum(df.index.duplicated())
    df = df.groupby(df.index).mean()
    
    # Print data checks
    print("Data checks:")
    print(f"1. Duplicated rows: {duplicates} ({round(duplicates / len(df) * 100, 2)}%) are dropped.")
    print(f"2. Duplicated indexes: {index_duplicates} ({round(100 * index_duplicates / len(df), 2)}%) are averaged.")
    print(f"3. Data's duration is {(df.index[-1] - df.index[0]).total_seconds() / (365 * 24 * 3600)} years.")
    
    # Check completeness
    all_timestamps = pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")
    if len(all_timestamps) != len(df):
        missing_timestamps = len(all_timestamps.difference(df.index))
        df = df.resample('min').asfreq().ffill()
        print(f"4. There are {missing_timestamps} ({round(100 * missing_timestamps / len(df), 2)}%) missing candles. Filled using 'forward fill' method.")
    else:
        print("Data is complete.")
    
    return df


def save_to_csv(df: pd.DataFrame, symbol_name: str) -> None:
    """Save the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Kline DataFrame object.
        symbol_name (str): Symbol name.
    """
    dirname = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.join(dirname, "../data", f"{symbol_name}.csv")
    df.to_csv(filename)
    print(f"Data saved to {filename}")


def main(symbol_name: str) -> None:
    """Main function to process symbol data.

    Args:
        symbol_name (str): Target symbol name.
    """
    try:
        db = connection()
        symbol = fetch_symbol(db, symbol_name)
        
        if not symbol:
            print(f"Symbol '{symbol_name}' not found. Please use a valid symbol.")
            return
        
        klines = fetch_klines(db, symbol.id)
        print(f"Data for symbol '{symbol.symbol}' is loaded.")
        
        df = process_klines(klines)
        print("Dataframe is created.")
        
        df = clean_data(df)
        
        print(df.info())
        print(df.describe())
        
        save_to_csv(df, symbol.symbol)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        db.close()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <symbol>")
        sys.exit(1)
    
    main(sys.argv[1])
    