"""
Интегрированная алгоритмическая торговая система
Объединяет все компоненты:
- Брокерские API (MT5, IB, Binance)
- Машинное обучение и оптимизация
- Анализ настроений рынка
- Портфельная оптимизация
- Комплексный бэктестинг
- Единая система управления
"""

import asyncio
import logging
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# Импорт  модулей
try:
    from broker_api_integration import (
        BrokerAPI, MetaTrader5API, InteractiveBrokersAPI, BinanceAPI,
        OrderType, OrderSide, Position, Order
    )
    BROKER_API_AVAILABLE = True
except ImportError:
    BROKER_API_AVAILABLE = False
    logging.warning("Модуль broker_api_integration не найден")

try:
    from ml_optimization import (
        MLOptimizer, ParameterOptimizer, FeatureEngineer,
        OptimizationConfig, OptimizationResults
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Модуль ml_optimization не найден")

try:
    from sentiment_analysis import (
        SentimentAggregator, SentimentConfig, AggregatedSentiment
    )
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logging.warning("Модуль sentiment_analysis не найден")

try:
    from portfolio_optimization import (
        PortfolioOptimizer, PortfolioConfig, PortfolioAnalyzer,
        CorrelationManager, PortfolioMetrics
    )
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False
    logging.warning("Модуль portfolio_optimization не найден")

try:
    from backtesting_system import (
        BacktestEngine, BacktestConfig, BacktestResults,
        MonteCarloAnalyzer, WalkForwardAnalyzer, BacktestReporter
    )
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    logging.warning("Модуль backtesting_system не найден")

# Базовые классы для fallback
if not BROKER_API_AVAILABLE:
    class BrokerAPI(ABC):
        pass
    
    class OrderType:
        MARKET = "market"
        LIMIT = "limit"
    
    class OrderSide:
        BUY = "buy"
        SELL = "sell"

@dataclass
class TradingSystemConfig:
    """Конфигурация торговой системы"""
    # Общие настройки
    system_name: str = "Advanced Trading System"
    trading_mode: str = "live"  # live, paper, backtest
    
    # Активы и рынки
    forex_symbols: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    crypto_symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
    
    # Временные рамки
    primary_timeframe: str = "1h"
    secondary_timeframes: List[str] = field(default_factory=lambda: ['4h', '1d'])
    
    # Управление рисками
    max_portfolio_risk: float = 0.02  # 2% от капитала
    max_correlation: float = 0.7
    max_positions: int = 10
    
    # Настройки компонентов
    use_ml_optimization: bool = True
    use_sentiment_analysis: bool = True
    use_portfolio_optimization: bool = True
    
    # Интервалы обновления
    signal_update_interval: int = 300  # секунды
    portfolio_rebalance_interval: int = 3600  # секунды
    sentiment_update_interval: int = 600  # секунды
    
    # API ключи (должны быть в переменных окружения)
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    
    twitter_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None

@dataclass
class TradingSignal:
    """Торговый сигнал"""
    symbol: str
    side: str  # buy, sell, hold
    confidence: float  # 0-1
    strength: float  # 0-1
    timeframe: str
    timestamp: datetime
    source: str  # technical, sentiment, ml, portfolio
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemState:
    """Состояние системы"""
    is_running: bool = False
    last_update: Optional[datetime] = None
    active_positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: List[Order] = field(default_factory=list)
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

class SignalGenerator(ABC):
    """Абстрактный генератор сигналов"""
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Генерация торговых сигналов"""
        pass

class TechnicalSignalGenerator(SignalGenerator):
    """Генератор технических сигналов"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.logger = logging.getLogger("TechnicalSignalGenerator")
    
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Генерация технических сигналов"""
        signals = []
        
        for symbol, df in market_data.items():
            try:
                if len(df) < 50:
                    continue
                
                # Простые технические индикаторы
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['rsi'] = self._calculate_rsi(df['close'])
                
                current_price = df['close'].iloc[-1]
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                rsi = df['rsi'].iloc[-1]
                
                # Логика сигналов
                if current_price > sma_20 > sma_50 and rsi < 70:
                    # Бычий сигнал
                    confidence = min(0.8, (current_price - sma_20) / sma_20 * 10)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        side="buy",
                        confidence=confidence,
                        strength=0.6,
                        timeframe=self.config.primary_timeframe,
                        timestamp=datetime.now(),
                        source="technical",
                        metadata={"rsi": rsi, "price_vs_sma": current_price/sma_20}
                    ))
                
                elif current_price < sma_20 < sma_50 and rsi > 30:
                    # Медвежий сигнал
                    confidence = min(0.8, (sma_20 - current_price) / sma_20 * 10)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        side="sell",
                        confidence=confidence,
                        strength=0.6,
                        timeframe=self.config.primary_timeframe,
                        timestamp=datetime.now(),
                        source="technical",
                        metadata={"rsi": rsi, "price_vs_sma": current_price/sma_20}
                    ))
                
            except Exception as e:
                self.logger.error(f"Ошибка генерации технических сигналов для {symbol}: {e}")
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class MLSignalGenerator(SignalGenerator):
    """Генератор ML сигналов"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.ml_optimizer = None
        self.logger = logging.getLogger("MLSignalGenerator")
        
        if ML_AVAILABLE:
            ml_config = OptimizationConfig()
            self.ml_optimizer = MLOptimizer(ml_config)
    
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Генерация ML сигналов"""
        signals = []
        
        if not self.ml_optimizer:
            return signals
        
        for symbol, df in market_data.items():
            try:
                if len(df) < 100:
                    continue
                
                # Подготовка данных для ML
                features = self._prepare_features(df)
                
                if len(features) < 50:
                    continue
                
                # Предсказание направления движения
                prediction = self.ml_optimizer.predict_direction(features)
                
                if prediction is not None:
                    confidence = abs(prediction)
                    side = "buy" if prediction > 0.1 else "sell" if prediction < -0.1 else "hold"
                    
                    if side != "hold":
                        signals.append(TradingSignal(
                            symbol=symbol,
                            side=side,
                            confidence=confidence,
                            strength=0.7,
                            timeframe=self.config.primary_timeframe,
                            timestamp=datetime.now(),
                            source="ml",
                            metadata={"prediction": prediction}
                        ))
                
            except Exception as e:
                self.logger.error(f"Ошибка генерации ML сигналов для {symbol}: {e}")
        
        return signals
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для ML"""
        features = df.copy()
        
        # Технические индикаторы
        features['returns'] = features['close'].pct_change()
        features['sma_10'] = features['close'].rolling(10).mean()
        features['sma_20'] = features['close'].rolling(20).mean()
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
        
        return features.dropna()

class SentimentSignalGenerator(SignalGenerator):
    """Генератор сигналов на основе настроений"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.sentiment_aggregator = None
        self.logger = logging.getLogger("SentimentSignalGenerator")
        
        if SENTIMENT_AVAILABLE:
            sentiment_config = SentimentConfig()
            self.sentiment_aggregator = SentimentAggregator(sentiment_config)
    
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Генерация сигналов на основе настроений"""
        signals = []
        
        if not self.sentiment_aggregator:
            return signals
        
        try:
            symbols = list(market_data.keys())
            sentiment_results = await self.sentiment_aggregator.analyze_sentiment_for_symbols(symbols)
            
            for symbol, sentiment_data in sentiment_results.items():
                if sentiment_data.confidence > 0.3:  # Минимальная уверенность
                    
                    if sentiment_data.overall_sentiment > 0.2:
                        side = "buy"
                    elif sentiment_data.overall_sentiment < -0.2:
                        side = "sell"
                    else:
                        continue
                    
                    signals.append(TradingSignal(
                        symbol=symbol,
                        side=side,
                        confidence=sentiment_data.confidence,
                        strength=abs(sentiment_data.overall_sentiment),
                        timeframe=self.config.primary_timeframe,
                        timestamp=datetime.now(),
                        source="sentiment",
                        metadata={
                            "sentiment_score": sentiment_data.overall_sentiment,
                            "trend": sentiment_data.trend,
                            "volume": sentiment_data.volume
                        }
                    ))
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации сентимент сигналов: {e}")
        
        return signals

class SignalAggregator:
    """Агрегатор сигналов"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.logger = logging.getLogger("SignalAggregator")
    
    def aggregate_signals(self, signals: List[TradingSignal]) -> Dict[str, TradingSignal]:
        """Агрегация сигналов по символам"""
        
        aggregated = {}
        
        # Группировка по символам
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Агрегация для каждого символа
        for symbol, symbol_signal_list in symbol_signals.items():
            
            # Взвешенное голосование
            buy_score = 0
            sell_score = 0
            total_weight = 0
            
            weights = {
                "technical": 0.4,
                "ml": 0.4,
                "sentiment": 0.2,
                "portfolio": 0.3
            }
            
            for signal in symbol_signal_list:
                weight = weights.get(signal.source, 0.1)
                score = signal.confidence * signal.strength * weight
                
                if signal.side == "buy":
                    buy_score += score
                elif signal.side == "sell":
                    sell_score += score
                
                total_weight += weight
            
            # Определение итогового сигнала
            if total_weight > 0:
                net_score = (buy_score - sell_score) / total_weight
                
                if abs(net_score) > 0.1:  # Минимальный порог
                    side = "buy" if net_score > 0 else "sell"
                    confidence = min(1.0, abs(net_score))
                    strength = min(1.0, max(buy_score, sell_score) / total_weight)
                    
                    aggregated[symbol] = TradingSignal(
                        symbol=symbol,
                        side=side,
                        confidence=confidence,
                        strength=strength,
                        timeframe=self.config.primary_timeframe,
                        timestamp=datetime.now(),
                        source="aggregated",
                        metadata={
                            "buy_score": buy_score,
                            "sell_score": sell_score,
                            "net_score": net_score,
                            "signal_count": len(symbol_signal_list)
                        }
                    )
        
        return aggregated

class RiskManager:
    """Менеджер рисков"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.logger = logging.getLogger("RiskManager")
    
    def validate_signal(self, signal: TradingSignal, 
                       current_positions: Dict[str, Position],
                       portfolio_value: float) -> bool:
        """Валидация сигнала с точки зрения рисков"""
        
        try:
            # Проверка максимального количества позиций
            if len(current_positions) >= self.config.max_positions and signal.symbol not in current_positions:
                self.logger.warning(f"Достигнуто максимальное количество позиций: {self.config.max_positions}")
                return False
            
            # Проверка максимального риска портфеля
            current_risk = self._calculate_portfolio_risk(current_positions, portfolio_value)
            if current_risk > self.config.max_portfolio_risk:
                self.logger.warning(f"Превышен максимальный риск портфеля: {current_risk:.2%} > {self.config.max_portfolio_risk:.2%}")
                return False
            
            # Проверка корреляций (упрощенная)
            if self._check_correlation_risk(signal.symbol, current_positions):
                self.logger.warning(f"Высокая корреляция с существующими позициями: {signal.symbol}")
                return False
            
            # Проверка минимальной уверенности
            if signal.confidence < 0.3:
                self.logger.debug(f"Низкая уверенность сигнала: {signal.confidence:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации сигнала: {e}")
            return False
    
    def _calculate_portfolio_risk(self, positions: Dict[str, Position], portfolio_value: float) -> float:
        """Расчет риска портфеля"""
        if portfolio_value <= 0:
            return 0.0
        
        total_exposure = sum(abs(pos.quantity * pos.avg_price) for pos in positions.values())
        return total_exposure / portfolio_value
    
    def _check_correlation_risk(self, symbol: str, positions: Dict[str, Position]) -> bool:
        """Проверка корреляционного риска"""
        # Упрощенная проверка на основе типа актива
        symbol_type = self._get_asset_type(symbol)
        
        same_type_count = 0
        for pos_symbol in positions.keys():
            if self._get_asset_type(pos_symbol) == symbol_type:
                same_type_count += 1
        
        # Не более 70% позиций одного типа
        max_same_type = max(1, int(self.config.max_positions * 0.7))
        return same_type_count >= max_same_type
    
    def _get_asset_type(self, symbol: str) -> str:
        """Определение типа актива"""
        if any(pair in symbol.upper() for pair in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']):
            return "forex"
        elif any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI']):
            return "crypto"
        else:
            return "other"

class TradingSystem:
    """Главная торговая система"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.state = SystemState()
        self.logger = logging.getLogger("TradingSystem")
        
        # Компоненты системы
        self.signal_generators = []
        self.signal_aggregator = SignalAggregator(config)
        self.risk_manager = RiskManager(config)
        
        # Брокерские API
        self.brokers = {}
        
        # Портфельный оптимизатор
        self.portfolio_optimizer = None
        if PORTFOLIO_AVAILABLE:
            portfolio_config = PortfolioConfig()
            self.portfolio_optimizer = PortfolioOptimizer(portfolio_config)
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Потоки для асинхронной работы
        self.signal_thread = None
        self.portfolio_thread = None
        self.monitoring_thread = None
    
    def _initialize_components(self):
        """Инициализация компонентов"""
        
        # Генераторы сигналов
        self.signal_generators.append(TechnicalSignalGenerator(self.config))
        
        if self.config.use_ml_optimization and ML_AVAILABLE:
            self.signal_generators.append(MLSignalGenerator(self.config))
        
        if self.config.use_sentiment_analysis and SENTIMENT_AVAILABLE:
            self.signal_generators.append(SentimentSignalGenerator(self.config))
        
        # Брокерские API
        if BROKER_API_AVAILABLE:
            try:
                if self.config.mt5_login:
                    self.brokers['mt5'] = MetaTrader5API(
                        self.config.mt5_login,
                        self.config.mt5_password,
                        self.config.mt5_server
                    )
                
                if self.config.binance_api_key:
                    self.brokers['binance'] = BinanceAPI(
                        self.config.binance_api_key,
                        self.config.binance_secret_key
                    )
                
                # Interactive Brokers требует TWS/Gateway
                # self.brokers['ib'] = InteractiveBrokersAPI(...)
                
            except Exception as e:
                self.logger.error(f"Ошибка инициализации брокерских API: {e}")
    
    async def start(self):
        """Запуск торговой системы"""
        
        self.logger.info("🚀 Запуск алгоритмической торговой системы")
        self.logger.info(f"   Режим: {self.config.trading_mode}")
        self.logger.info(f"   Forex символы: {self.config.forex_symbols}")
        self.logger.info(f"   Crypto символы: {self.config.crypto_symbols}")
        
        self.state.is_running = True
        self.state.last_update = datetime.now()
        
        try:
            # Подключение к брокерам
            await self._connect_brokers()
            
            # Запуск основных потоков
            self._start_threads()
            
            # Основной цикл
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Ошибка в главном цикле: {e}")
            self.state.last_error = str(e)
            self.state.error_count += 1
        
        finally:
            await self.stop()
    
    async def _connect_brokers(self):
        """Подключение к брокерам"""
        
        for broker_name, broker_api in self.brokers.items():
            try:
                await broker_api.connect()
                self.logger.info(f"✅ Подключен к {broker_name}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка подключения к {broker_name}: {e}")
    
    def _start_threads(self):
        """Запуск рабочих потоков"""
        
        # Поток генерации сигналов
        self.signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        self.signal_thread.start()
        
        # Поток портфельной оптимизации
        if self.config.use_portfolio_optimization:
            self.portfolio_thread = threading.Thread(target=self._portfolio_loop, daemon=True)
            self.portfolio_thread.start()
        
        # Поток мониторинга
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    async def _main_loop(self):
        """Главный цикл системы"""
        
        while self.state.is_running:
            try:
                # Обновление состояния позиций
                await self._update_positions()
                
                # Обновление состояния портфеля
                await self._update_portfolio_state()
                
                # Пауза
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Ошибка в главном цикле: {e}")
                self.state.error_count += 1
                await asyncio.sleep(30)
    
    def _signal_loop(self):
        """Цикл генерации и обработки сигналов"""
        
        while self.state.is_running:
            try:
                # Получение рыночных данных
                market_data = self._get_market_data()
                
                if market_data:
                    # Генерация сигналов
                    all_signals = []
                    
                    for generator in self.signal_generators:
                        try:
                            signals = asyncio.run(generator.generate_signals(market_data))
                            all_signals.extend(signals)
                        except Exception as e:
                            self.logger.error(f"Ошибка генерации сигналов {generator.__class__.__name__}: {e}")
                    
                    # Агрегация сигналов
                    aggregated_signals = self.signal_aggregator.aggregate_signals(all_signals)
                    
                    # Обработка сигналов
                    for symbol, signal in aggregated_signals.items():
                        if self.risk_manager.validate_signal(signal, self.state.active_positions, self.state.portfolio_value):
                            asyncio.run(self._execute_signal(signal))
                
                time.sleep(self.config.signal_update_interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле сигналов: {e}")
                time.sleep(60)
    
    def _portfolio_loop(self):
        """Цикл портфельной оптимизации"""
        
        while self.state.is_running:
            try:
                if self.portfolio_optimizer and len(self.state.active_positions) > 1:
                    # Получение исторических данных
                    symbols = list(self.state.active_positions.keys())
                    returns_data = self._get_returns_data(symbols)
                    
                    if returns_data is not None and len(returns_data) > 50:
                        # Оптимизация портфеля
                        optimal_weights = self.portfolio_optimizer.optimize_portfolio(returns_data)
                        
                        # Ребалансировка при необходимости
                        if self._should_rebalance(optimal_weights):
                            asyncio.run(self._rebalance_portfolio(optimal_weights))
                
                time.sleep(self.config.portfolio_rebalance_interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле портфеля: {e}")
                time.sleep(300)
    
    def _monitoring_loop(self):
        """Цикл мониторинга системы"""
        
        while self.state.is_running:
            try:
                # Логирование состояния
                self.logger.info(f"📊 Состояние системы:")
                self.logger.info(f"   Активные позиции: {len(self.state.active_positions)}")
                self.logger.info(f"   Стоимость портфеля: ${self.state.portfolio_value:,.2f}")
                self.logger.info(f"   Дневная P&L: ${self.state.daily_pnl:,.2f}")
                self.logger.info(f"   Общая P&L: ${self.state.total_pnl:,.2f}")
                
                if self.state.error_count > 0:
                    self.logger.warning(f"   Ошибки: {self.state.error_count}")
                
                time.sleep(300)  # Каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге: {e}")
                time.sleep(60)
    
    def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Получение рыночных данных"""
        
        market_data = {}
        all_symbols = self.config.forex_symbols + self.config.crypto_symbols
        
        for symbol in all_symbols:
            try:
                # Попытка получить данные от брокеров
                for broker_name, broker_api in self.brokers.items():
                    try:
                        df = broker_api.get_historical_data(symbol, self.config.primary_timeframe, 100)
                        if df is not None and len(df) > 0:
                            market_data[symbol] = df
                            break
                    except Exception as e:
                        self.logger.debug(f"Не удалось получить данные {symbol} от {broker_name}: {e}")
                
                # Fallback - генерация тестовых данных
                if symbol not in market_data:
                    market_data[symbol] = self._generate_test_data(symbol)
                
            except Exception as e:
                self.logger.error(f"Ошибка получения данных для {symbol}: {e}")
        
        return market_data
    
    def _generate_test_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Генерация тестовых данных"""
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Симуляция цен
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0, 0.01, periods)
        prices = 100 * (1 + pd.Series(returns)).cumprod()
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(prices.iloc[0]),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)
    
    async def _execute_signal(self, signal: TradingSignal):
        """Исполнение торгового сигнала"""
        
        try:
            self.logger.info(f"🎯 Исполнение сигнала: {signal.symbol} {signal.side} (уверенность: {signal.confidence:.2f})")
            
            # Определение размера позиции
            position_size = self._calculate_position_size(signal)
            
            # Выбор подходящего брокера
            broker = self._select_broker(signal.symbol)
            
            if broker:
                # Создание ордера
                order = Order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY if signal.side == "buy" else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position_size
                )
                
                # Отправка ордера
                result = await broker.place_order(order)
                
                if result:
                    self.logger.info(f"✅ Ордер размещен: {order.symbol} {order.side} {order.quantity}")
                    self.state.pending_orders.append(order)
                else:
                    self.logger.error(f"❌ Ошибка размещения ордера: {order.symbol}")
            
            else:
                self.logger.warning(f"Нет доступного брокера для {signal.symbol}")
        
        except Exception as e:
            self.logger.error(f"Ошибка исполнения сигнала: {e}")
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Расчет размера позиции"""
        
        # Упрощенный расчет на основе риска
        risk_amount = self.state.portfolio_value * self.config.max_portfolio_risk
        
        # Корректировка на уверенность сигнала
        adjusted_risk = risk_amount * signal.confidence
        
        # Примерный размер позиции (нужна текущая цена для точного расчета)
        return max(0.01, adjusted_risk / 1000)  # Упрощенный расчет
    
    def _select_broker(self, symbol: str) -> Optional[BrokerAPI]:
        """Выбор подходящего брокера для символа"""
        
        # Логика выбора брокера на основе типа актива
        if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA']):
            return self.brokers.get('binance')
        else:
            return self.brokers.get('mt5') or self.brokers.get('ib')
    
    async def _update_positions(self):
        """Обновление состояния позиций"""
        
        for broker_name, broker_api in self.brokers.items():
            try:
                positions = await broker_api.get_positions()
                
                for position in positions:
                    self.state.active_positions[position.symbol] = position
                
            except Exception as e:
                self.logger.error(f"Ошибка обновления позиций от {broker_name}: {e}")
    
    async def _update_portfolio_state(self):
        """Обновление состояния портфеля"""
        
        try:
            total_value = 0
            total_pnl = 0
            
            for position in self.state.active_positions.values():
                total_value += abs(position.quantity * position.avg_price)
                total_pnl += position.unrealized_pnl + position.realized_pnl
            
            self.state.portfolio_value = total_value
            self.state.total_pnl = total_pnl
            self.state.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления состояния портфеля: {e}")
    
    def _get_returns_data(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Получение данных доходностей для портфельной оптимизации"""
        
        try:
            returns_data = {}
            
            for symbol in symbols:
                market_data = self._get_market_data()
                if symbol in market_data:
                    df = market_data[symbol]
                    returns = df['close'].pct_change().dropna()
                    returns_data[symbol] = returns
            
            if returns_data:
                return pd.DataFrame(returns_data).dropna()
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных доходностей: {e}")
        
        return None
    
    def _should_rebalance(self, optimal_weights: Dict[str, float]) -> bool:
        """Проверка необходимости ребалансировки"""
        
        # Упрощенная логика - ребалансировка при значительном отклонении
        current_weights = self._calculate_current_weights()
        
        for symbol, optimal_weight in optimal_weights.items():
            current_weight = current_weights.get(symbol, 0)
            if abs(optimal_weight - current_weight) > 0.05:  # 5% отклонение
                return True
        
        return False
    
    def _calculate_current_weights(self) -> Dict[str, float]:
        """Расчет текущих весов портфеля"""
        
        total_value = sum(abs(pos.quantity * pos.avg_price) for pos in self.state.active_positions.values())
        
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self.state.active_positions.items():
            weights[symbol] = abs(position.quantity * position.avg_price) / total_value
        
        return weights
    
    async def _rebalance_portfolio(self, optimal_weights: Dict[str, float]):
        """Ребалансировка портфеля"""
        
        self.logger.info("🔄 Начало ребалансировки портфеля")
        
        try:
            current_weights = self._calculate_current_weights()
            
            for symbol, target_weight in optimal_weights.items():
                current_weight = current_weights.get(symbol, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.05:  # Значимое отклонение
                    # Расчет необходимого изменения позиции
                    target_value = self.state.portfolio_value * target_weight
                    current_value = self.state.portfolio_value * current_weight
                    value_diff = target_value - current_value
                    
                    # Создание сигнала для корректировки
                    side = "buy" if value_diff > 0 else "sell"
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        side=side,
                        confidence=0.8,
                        strength=0.5,
                        timeframe=self.config.primary_timeframe,
                        timestamp=datetime.now(),
                        source="portfolio",
                        metadata={"rebalance": True, "target_weight": target_weight}
                    )
                    
                    await self._execute_signal(signal)
            
            self.logger.info("✅ Ребалансировка завершена")
            
        except Exception as e:
            self.logger.error(f"Ошибка ребалансировки: {e}")
    
    async def stop(self):
        """Остановка торговой системы"""
        
        self.logger.info("🛑 Остановка торговой системы")
        
        self.state.is_running = False
        
        # Отключение от брокеров
        for broker_name, broker_api in self.brokers.items():
            try:
                await broker_api.disconnect()
                self.logger.info(f"✅ Отключен от {broker_name}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка отключения от {broker_name}: {e}")
        
        self.logger.info("✅ Торговая система остановлена")
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        
        return {
            "is_running": self.state.is_running,
            "last_update": self.state.last_update,
            "active_positions": len(self.state.active_positions),
            "pending_orders": len(self.state.pending_orders),
            "portfolio_value": self.state.portfolio_value,
            "daily_pnl": self.state.daily_pnl,
            "total_pnl": self.state.total_pnl,
            "error_count": self.state.error_count,
            "last_error": self.state.last_error,
            "brokers_connected": list(self.brokers.keys()),
            "signal_generators": len(self.signal_generators)
        }

# Демонстрация интегрированной системы
async def demo_integrated_system():
    """Демонстрация интегрированной торговой системы"""
    
    print("🚀 ДЕМОНСТРАЦИЯ ИНТЕГРИРОВАННОЙ ТОРГОВОЙ СИСТЕМЫ")
    print("=" * 70)
    
    # Конфигурация системы
    config = TradingSystemConfig(
        system_name="Demo Advanced Trading System",
        trading_mode="paper",  # Бумажная торговля для демонстрации
        forex_symbols=['EURUSD', 'GBPUSD'],
        crypto_symbols=['BTCUSDT', 'ETHUSDT'],
        use_ml_optimization=ML_AVAILABLE,
        use_sentiment_analysis=SENTIMENT_AVAILABLE,
        use_portfolio_optimization=PORTFOLIO_AVAILABLE,
        signal_update_interval=60,  # Быстрее для демонстрации
        portfolio_rebalance_interval=300
    )
    
    print(f"📋 Конфигурация системы:")
    print(f"   Название: {config.system_name}")
    print(f"   Режим: {config.trading_mode}")
    print(f"   Forex символы: {config.forex_symbols}")
    print(f"   Crypto символы: {config.crypto_symbols}")
    print(f"   ML оптимизация: {config.use_ml_optimization}")
    print(f"   Анализ настроений: {config.use_sentiment_analysis}")
    print(f"   Портфельная оптимизация: {config.use_portfolio_optimization}")
    
    # Создание торговой системы
    trading_system = TradingSystem(config)
    
    print(f"\n🔧 Инициализация компонентов...")
    print(f"   Генераторы сигналов: {len(trading_system.signal_generators)}")
    print(f"   Подключенные брокеры: {len(trading_system.brokers)}")
    
    try:
        # Демонстрация генерации сигналов
        print(f"\n🎯 Демонстрация генерации сигналов...")
        
        market_data = trading_system._get_market_data()
        print(f"   Получены данные для {len(market_data)} символов")
        
        all_signals = []
        for generator in trading_system.signal_generators:
            try:
                signals = await generator.generate_signals(market_data)
                all_signals.extend(signals)
                print(f"   {generator.__class__.__name__}: {len(signals)} сигналов")
            except Exception as e:
                print(f"   ❌ Ошибка в {generator.__class__.__name__}: {e}")
        
        # Агрегация сигналов
        aggregated_signals = trading_system.signal_aggregator.aggregate_signals(all_signals)
        print(f"   Агрегированные сигналы: {len(aggregated_signals)}")
        
        for symbol, signal in aggregated_signals.items():
            print(f"     {symbol}: {signal.side} (уверенность: {signal.confidence:.2f}, сила: {signal.strength:.2f})")
        
        # Демонстрация валидации рисков
        print(f"\n⚠️ Демонстрация управления рисками...")
        
        valid_signals = 0
        for signal in aggregated_signals.values():
            if trading_system.risk_manager.validate_signal(signal, {}, 100000):
                valid_signals += 1
        
        print(f"   Прошли валидацию: {valid_signals}/{len(aggregated_signals)} сигналов")
        
        # Демонстрация портфельной оптимизации
        if trading_system.portfolio_optimizer and len(market_data) > 1:
            print(f"\n📊 Демонстрация портфельной оптимизации...")
            
            # Подготовка данных доходностей
            returns_data = {}
            for symbol, df in market_data.items():
                returns_data[symbol] = df['close'].pct_change().dropna()
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if len(returns_df) > 20:
                optimal_weights = trading_system.portfolio_optimizer.optimize_portfolio(returns_df)
                print(f"   Оптимальные веса портфеля:")
                for symbol, weight in optimal_weights.items():
                    print(f"     {symbol}: {weight:.1%}")
        
        # Статус системы
        status = trading_system.get_status()
        print(f"\n📈 Статус системы:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск демонстрации
    asyncio.run(demo_integrated_system())

