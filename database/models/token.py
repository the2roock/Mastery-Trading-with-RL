import pendulum

from sqlalchemy import Column, Integer, Double, String, JSON, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship

try:
    from Base import Base
except ImportError:
    from ..Base import Base


class Symbol(Base):
    __tablename__ = "symbol"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    data = Column(JSON, default=None)
    time_create = Column(TIMESTAMP(timezone=True))
    time_update = Column(TIMESTAMP(timezone=True))
    status = Column(Integer)

    klines = relationship("Kline", backref="symbol")
    order_book = relationship("OrderBook", backref="symbol")

    def __init__(self, *args, **kwargs):
        super(Symbol, self).__init__(*args, **kwargs)
        self.time_create = pendulum.now("UTC")
        self.time_update = pendulum.now("UTC")

    def __repr__(self):
        return f"<Symbol: " \
               f"id={self.id}, " \
               f"symbol={self.symbol}, " \
               f"data={self.data}, " \
               f"time_create={self.time_create}, " \
               f"time_update={self.time_update}>"


class Kline(Base):
    __tablename__ = "kline"

    id = Column(Integer, primary_key=True)
    id_symbol = Column(Integer, ForeignKey("symbol.id"), index=True)
    open = Column(Double)
    high = Column(Double)
    low = Column(Double)
    close = Column(Double)
    volume = Column(Double)
    number_of_trades = Column(Integer)
    time_open = Column(TIMESTAMP, index=True)
    time_close = Column(TIMESTAMP)
    time_create = Column(TIMESTAMP(timezone=True))
    time_update = Column(TIMESTAMP(timezone=True))

    def __init__(self, *args, **kwargs):
        super(Kline, self).__init__(*args, **kwargs)
        self.time_create = pendulum.now("UTC")
        self.time_update = pendulum.now("UTC")
        
    def __repr__(self):
        return f"<Kline: " \
               f"id={self.id}, " \
               f"id_symbol={self.id_symbol}, " \
               f"open={self.open}, " \
               f"high={self.high}, " \
               f"low={self.low}, " \
               f"close={self.close}, " \
               f"volume={self.volume}, " \
               f"number_of_trades={self.number_of_trades}, " \
               f"time_open={self.time_open}, " \
               f"time_close={self.time_close}, " \
               f"time_create={self.time_create}, " \
               f"time_update={self.time_update}>"


class OrderBook(Base):
    __tablename__ = "order_book"

    id = Column(Integer, primary_key=True)
    id_symbol = Column(Integer, ForeignKey("symbol.id"), index=True)
    ask = Column(Text)
    bid = Column(Text)
    time_open = Column(TIMESTAMP, index=True)
    time_create = Column(TIMESTAMP(timezone=True))
    time_update = Column(TIMESTAMP(timezone=True))

    def __init__(self, *args, **kwargs):
        super(OrderBook, self).__init__(*args, **kwargs)
        self.time_create = pendulum.now("UTC")
        self.time_update = pendulum.now("UTC")

    def __repr__(self):
        return f"<OrderBook: " \
               f"id={self.id}, " \
               f"id_symbol={self.id_symbol}, " \
               f"ask={self.ask}, " \
               f"bid={self.bid}, " \
               f"time_open={self.time_open}, " \
               f"time_create={self.time_create}, " \
               f"time_update={self.time_update}>"
