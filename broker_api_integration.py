"""
Модуль интеграции с брокерскими API
Поддерживаемые брокеры:
- MetaTrader 5 (Forex/CFD)
- Interactive Brokers (Stocks/Forex/Options)
- Binance (Cryptocurrency)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Попытка импорта API библиотек (с обработкой отсутствующих зависимостей)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 не установлен. Установите: pip install MetaTrader5")

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logging.warning("Interactive Brokers API не установлен. Установите: pip install ibapi")

try:
    from binance.client import Client as BinanceClient
    from binance.enums import *
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logging.warning("Binance API не установлен. Установите: pip install python-binance")

@dataclass
class BrokerConfig:
    """Конфигурация брокера"""
    broker_type: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    server: Optional[str] = None
    account: Optional[str] = None
    sandbox: bool = True
    timeout: int = 30

@dataclass
class MarketData:
    """Рыночные данные"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    spread: float

@dataclass
class OrderRequest:
    """Запрос на размещение ордера"""
    symbol: str
    side: str  # 'buy' или 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # Good Till Cancelled

@dataclass
class OrderResponse:
    """Ответ на размещение ордера"""
    order_id: str
    status: str
    filled_quantity: float
    average_price: float
    commission: float
    timestamp: datetime

class BrokerAPI(ABC):
    """Абстрактный базовый класс для брокерских API"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.connected = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Подключение к брокеру"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Отключение от брокера"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Получение рыночных данных"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение исторических данных"""
        pass
    
    @abstractmethod
    async def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Размещение ордера"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Получение информации о счете"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение открытых позиций"""
        pass

class MetaTrader5API(BrokerAPI):
    """API для MetaTrader 5"""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 не установлен")
    
    async def connect(self) -> bool:
        """Подключение к MT5"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 инициализация не удалась: {mt5.last_error()}")
                return False
            
            # Авторизация если указаны данные
            if self.config.account and self.config.api_key:
                authorized = mt5.login(
                    login=int(self.config.account),
                    password=self.config.api_key,
                    server=self.config.server or ""
                )
                if not authorized:
                    self.logger.error(f"MT5 авторизация не удалась: {mt5.last_error()}")
                    return False
            
            self.connected = True
            self.logger.info("Успешное подключение к MT5")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к MT5: {e}")
            return False
    
    async def disconnect(self):
        """Отключение от MT5"""
        mt5.shutdown()
        self.connected = False
        self.logger.info("Отключение от MT5")
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Получение рыночных данных из MT5"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(tick.time),
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                spread=tick.ask - tick.bid
            )
        except Exception as e:
            self.logger.error(f"Ошибка получения данных MT5 для {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение исторических данных из MT5"""
        try:
            # Конвертация таймфрейма
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            if rates is None:
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Ошибка получения исторических данных MT5: {e}")
            return pd.DataFrame()
    
    async def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Размещение ордера в MT5"""
        try:
            # Получение информации о символе
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                self.logger.error(f"Символ {order.symbol} не найден")
                return None
            
            # Подготовка запроса
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.quantity,
                "type": mt5.ORDER_TYPE_BUY if order.side == 'buy' else mt5.ORDER_TYPE_SELL,
                "deviation": 20,
                "magic": 234000,
                "comment": "Algo Trading Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Добавление цены для лимитных ордеров
            if order.order_type == 'limit' and order.price:
                request["price"] = order.price
                request["type"] = mt5.ORDER_TYPE_BUY_LIMIT if order.side == 'buy' else mt5.ORDER_TYPE_SELL_LIMIT
            
            # Размещение ордера
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Ошибка размещения ордера MT5: {result.retcode}")
                return None
            
            return OrderResponse(
                order_id=str(result.order),
                status="filled" if result.retcode == mt5.TRADE_RETCODE_DONE else "pending",
                filled_quantity=result.volume,
                average_price=result.price,
                commission=0.0,  # MT5 не возвращает комиссию в ответе
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка размещения ордера MT5: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера в MT5"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            self.logger.error(f"Ошибка отмены ордера MT5: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Получение информации о счете MT5"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            return {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "currency": account_info.currency,
                "leverage": account_info.leverage
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о счете MT5: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение открытых позиций MT5"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.symbol,
                    "side": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                    "volume": pos.volume,
                    "open_price": pos.price_open,
                    "current_price": pos.price_current,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "comment": pos.comment
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка получения позиций MT5: {e}")
            return []

class BinanceAPI(BrokerAPI):
    """API для Binance"""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        if not BINANCE_AVAILABLE:
            raise ImportError("Binance API не установлен")
        self.client = None
    
    async def connect(self) -> bool:
        """Подключение к Binance"""
        try:
            if not self.config.api_key or not self.config.api_secret:
                self.logger.error("API ключи Binance не указаны")
                return False
            
            # Использование testnet для sandbox
            if self.config.sandbox:
                self.client = BinanceClient(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    testnet=True
                )
            else:
                self.client = BinanceClient(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret
                )
            
            # Проверка подключения
            account_info = self.client.get_account()
            self.connected = True
            self.logger.info("Успешное подключение к Binance")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к Binance: {e}")
            return False
    
    async def disconnect(self):
        """Отключение от Binance"""
        self.client = None
        self.connected = False
        self.logger.info("Отключение от Binance")
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Получение рыночных данных из Binance"""
        try:
            ticker = self.client.get_orderbook_ticker(symbol=symbol)
            price_ticker = self.client.get_symbol_ticker(symbol=symbol)
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=float(ticker['bidPrice']),
                ask=float(ticker['askPrice']),
                last=float(price_ticker['price']),
                volume=0.0,  # Получается отдельно
                spread=float(ticker['askPrice']) - float(ticker['bidPrice'])
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных Binance для {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение исторических данных из Binance"""
        try:
            # Конвертация таймфрейма
            tf_map = {
                'M1': Client.KLINE_INTERVAL_1MINUTE,
                'M5': Client.KLINE_INTERVAL_5MINUTE,
                'M15': Client.KLINE_INTERVAL_15MINUTE,
                'M30': Client.KLINE_INTERVAL_30MINUTE,
                'H1': Client.KLINE_INTERVAL_1HOUR,
                'H4': Client.KLINE_INTERVAL_4HOUR,
                'D1': Client.KLINE_INTERVAL_1DAY
            }
            
            interval = tf_map.get(timeframe, Client.KLINE_INTERVAL_1HOUR)
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date.strftime('%Y-%m-%d'),
                end_str=end_date.strftime('%Y-%m-%d')
            )
            
            if not klines:
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Конвертация в числовые типы
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Ошибка получения исторических данных Binance: {e}")
            return pd.DataFrame()
    
    async def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Размещение ордера в Binance"""
        try:
            # Подготовка параметров ордера
            order_params = {
                'symbol': order.symbol,
                'side': order.side.upper(),
                'type': order.order_type.upper(),
                'quantity': order.quantity,
                'timeInForce': order.time_in_force
            }
            
            # Добавление цены для лимитных ордеров
            if order.order_type.lower() == 'limit' and order.price:
                order_params['price'] = order.price
            
            # Размещение ордера
            result = self.client.create_order(**order_params)
            
            return OrderResponse(
                order_id=str(result['orderId']),
                status=result['status'].lower(),
                filled_quantity=float(result.get('executedQty', 0)),
                average_price=float(result.get('price', 0)),
                commission=0.0,  # Получается отдельно
                timestamp=datetime.fromtimestamp(result['transactTime'] / 1000)
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка размещения ордера Binance: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера в Binance"""
        try:
            result = self.client.cancel_order(orderId=int(order_id))
            return result['status'] == 'CANCELED'
            
        except Exception as e:
            self.logger.error(f"Ошибка отмены ордера Binance: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Получение информации о счете Binance"""
        try:
            account_info = self.client.get_account()
            
            balances = {}
            for balance in account_info['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    balances[balance['asset']] = {
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            
            return {
                "balances": balances,
                "can_trade": account_info['canTrade'],
                "can_withdraw": account_info['canWithdraw'],
                "can_deposit": account_info['canDeposit']
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о счете Binance: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение открытых позиций Binance (для фьючерсов)"""
        try:
            # Для спотовой торговли позиций нет, возвращаем балансы
            account_info = await self.get_account_info()
            positions = []
            
            for asset, balance in account_info.get('balances', {}).items():
                if balance['total'] > 0:
                    positions.append({
                        "symbol": asset,
                        "side": "long",  # Спотовые позиции всегда long
                        "volume": balance['total'],
                        "free": balance['free'],
                        "locked": balance['locked']
                    })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Ошибка получения позиций Binance: {e}")
            return []

class BrokerManager:
    """Менеджер для управления несколькими брокерами"""
    
    def __init__(self):
        self.brokers: Dict[str, BrokerAPI] = {}
        self.logger = logging.getLogger("BrokerManager")
    
    def add_broker(self, name: str, broker: BrokerAPI):
        """Добавление брокера"""
        self.brokers[name] = broker
        self.logger.info(f"Добавлен брокер: {name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Подключение ко всем брокерам"""
        results = {}
        for name, broker in self.brokers.items():
            try:
                results[name] = await broker.connect()
                if results[name]:
                    self.logger.info(f"Успешное подключение к {name}")
                else:
                    self.logger.error(f"Не удалось подключиться к {name}")
            except Exception as e:
                self.logger.error(f"Ошибка подключения к {name}: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self):
        """Отключение от всех брокеров"""
        for name, broker in self.brokers.items():
            try:
                await broker.disconnect()
                self.logger.info(f"Отключение от {name}")
            except Exception as e:
                self.logger.error(f"Ошибка отключения от {name}: {e}")
    
    def get_broker(self, name: str) -> Optional[BrokerAPI]:
        """Получение брокера по имени"""
        return self.brokers.get(name)
    
    async def get_all_market_data(self, symbol: str) -> Dict[str, MarketData]:
        """Получение рыночных данных от всех брокеров"""
        results = {}
        for name, broker in self.brokers.items():
            if broker.connected:
                try:
                    data = await broker.get_market_data(symbol)
                    if data:
                        results[name] = data
                except Exception as e:
                    self.logger.error(f"Ошибка получения данных от {name}: {e}")
        
        return results
    
    async def execute_order_with_best_price(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Исполнение ордера с лучшей ценой"""
        # Получение котировок от всех брокеров
        market_data = await self.get_all_market_data(order.symbol)
        
        if not market_data:
            self.logger.error("Нет доступных котировок")
            return None
        
        # Выбор лучшего брокера
        best_broker = None
        best_price = None
        
        for name, data in market_data.items():
            price = data.ask if order.side == 'buy' else data.bid
            
            if best_price is None or (
                (order.side == 'buy' and price < best_price) or
                (order.side == 'sell' and price > best_price)
            ):
                best_price = price
                best_broker = name
        
        if best_broker:
            broker = self.brokers[best_broker]
            self.logger.info(f"Исполнение ордера через {best_broker} по цене {best_price}")
            return await broker.place_order(order)
        
        return None

# Пример использования
async def demo_broker_integration():
    """Демонстрация интеграции с брокерами"""
    
    # Создание менеджера брокеров
    manager = BrokerManager()
    
    # Конфигурация MT5 (демо)
    mt5_config = BrokerConfig(
        broker_type="mt5",
        account="demo_account",
        api_key="demo_password",
        server="demo_server",
        sandbox=True
    )
    
    # Конфигурация Binance (тестнет)
    binance_config = BrokerConfig(
        broker_type="binance",
        api_key="your_testnet_api_key",
        api_secret="your_testnet_api_secret",
        sandbox=True
    )
    
    # Добавление брокеров
    if MT5_AVAILABLE:
        manager.add_broker("mt5", MetaTrader5API(mt5_config))
    
    if BINANCE_AVAILABLE:
        manager.add_broker("binance", BinanceAPI(binance_config))
    
    # Подключение
    connections = await manager.connect_all()
    print("Результаты подключения:", connections)
    
    # Получение рыночных данных
    if any(connections.values()):
        market_data = await manager.get_all_market_data("EURUSD")
        print("Рыночные данные:", market_data)
    
    # Отключение
    await manager.disconnect_all()

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Запуск демонстрации
    asyncio.run(demo_broker_integration())

