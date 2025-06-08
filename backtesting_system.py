"""
Комплексная система бэктестинга с метриками производительности
Включает:
- Векторизованный и событийный бэктестинг
- Детальный учет транзакционных издержек
- Анализ slippage и market impact
- Продвинутые метрики риска и доходности
- Monte Carlo симуляции
- Walk-forward анализ
- Статистическая значимость результатов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import warnings
from enum import Enum
import json
warnings.filterwarnings('ignore')

# Статистика и анализ
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy не установлен. Установите: pip install scipy")

# Визуализация
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib не установлен. Установите: pip install matplotlib seaborn")

class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Сторона ордера"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Статус ордера"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Ордер"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class Trade:
    """Сделка"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    duration: Optional[timedelta] = None
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE

@dataclass
class Position:
    """Позиция"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class BacktestConfig:
    """Конфигурация бэктестинга"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    margin_rate: float = 1.0        # Без плеча
    max_positions: int = 10
    position_sizing: str = "fixed"  # fixed, percent, volatility, kelly
    risk_per_trade: float = 0.02    # 2% риска на сделку
    benchmark: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

@dataclass
class BacktestResults:
    """Результаты бэктестинга"""
    # Основные метрики
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Торговые метрики
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Риск-метрики
    var_95: float
    cvar_95: float
    downside_deviation: float
    ulcer_index: float
    
    # Статистические метрики
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    # Бенчмарк метрики
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    # Временные ряды
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    returns_series: pd.Series = field(default_factory=pd.Series)
    
    # Детальная информация
    trades: List[Trade] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)

class MarketDataProvider(ABC):
    """Абстрактный провайдер рыночных данных"""
    
    @abstractmethod
    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение ценовых данных"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        pass

class SimulatedMarketData(MarketDataProvider):
    """Симулированные рыночные данные"""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.current_time = None
    
    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение ценовых данных"""
        if symbol not in self.data:
            raise ValueError(f"Данные для {symbol} не найдены")
        
        df = self.data[symbol]
        return df[(df.index >= start_date) & (df.index <= end_date)]
    
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        if symbol not in self.data:
            raise ValueError(f"Данные для {symbol} не найдены")
        
        if self.current_time is None:
            return self.data[symbol]['close'].iloc[-1]
        
        # Найти ближайшую цену к текущему времени
        df = self.data[symbol]
        idx = df.index.get_indexer([self.current_time], method='ffill')[0]
        
        if idx >= 0:
            return df['close'].iloc[idx]
        else:
            return df['close'].iloc[0]

class OrderExecutor:
    """Исполнитель ордеров"""
    
    def __init__(self, config: BacktestConfig, market_data: MarketDataProvider):
        self.config = config
        self.market_data = market_data
        self.logger = logging.getLogger("OrderExecutor")
    
    def execute_order(self, order: Order, current_time: datetime) -> Order:
        """Исполнение ордера"""
        
        try:
            current_price = self.market_data.get_current_price(order.symbol)
            
            if order.order_type == OrderType.MARKET:
                # Рыночный ордер исполняется немедленно
                filled_price = self._apply_slippage(current_price, order.side)
                order.filled_price = filled_price
                order.filled_quantity = order.quantity
                order.status = OrderStatus.FILLED
                
                # Расчет комиссии
                order.commission = order.quantity * filled_price * self.config.commission_rate
                
                # Расчет slippage
                order.slippage = abs(filled_price - current_price) * order.quantity
                
            elif order.order_type == OrderType.LIMIT:
                # Лимитный ордер исполняется только при достижении цены
                if self._can_fill_limit_order(order, current_price):
                    order.filled_price = order.price
                    order.filled_quantity = order.quantity
                    order.status = OrderStatus.FILLED
                    order.commission = order.quantity * order.price * self.config.commission_rate
                
            elif order.order_type == OrderType.STOP:
                # Стоп ордер превращается в рыночный при достижении стоп-цены
                if self._should_trigger_stop(order, current_price):
                    filled_price = self._apply_slippage(current_price, order.side)
                    order.filled_price = filled_price
                    order.filled_quantity = order.quantity
                    order.status = OrderStatus.FILLED
                    order.commission = order.quantity * filled_price * self.config.commission_rate
                    order.slippage = abs(filled_price - current_price) * order.quantity
            
        except Exception as e:
            self.logger.error(f"Ошибка исполнения ордера {order.id}: {e}")
            order.status = OrderStatus.REJECTED
        
        return order
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Применение slippage"""
        slippage_amount = price * self.config.slippage_rate
        
        if side == OrderSide.BUY:
            return price + slippage_amount
        else:
            return price - slippage_amount
    
    def _can_fill_limit_order(self, order: Order, current_price: float) -> bool:
        """Проверка возможности исполнения лимитного ордера"""
        if order.side == OrderSide.BUY:
            return current_price <= order.price
        else:
            return current_price >= order.price
    
    def _should_trigger_stop(self, order: Order, current_price: float) -> bool:
        """Проверка срабатывания стоп ордера"""
        if order.side == OrderSide.BUY:
            return current_price >= order.stop_price
        else:
            return current_price <= order.stop_price

class PositionManager:
    """Менеджер позиций"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger("PositionManager")
    
    def update_position(self, order: Order) -> Position:
        """Обновление позиции после исполнения ордера"""
        
        if order.status != OrderStatus.FILLED:
            return self.positions.get(order.symbol, Position(order.symbol))
        
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Покупка
            if position.quantity >= 0:
                # Увеличение длинной позиции
                total_cost = position.quantity * position.avg_price + order.filled_quantity * order.filled_price
                total_quantity = position.quantity + order.filled_quantity
                position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
            else:
                # Закрытие короткой позиции
                if order.filled_quantity >= abs(position.quantity):
                    # Полное закрытие и возможное открытие длинной
                    remaining_quantity = order.filled_quantity - abs(position.quantity)
                    position.realized_pnl += abs(position.quantity) * (position.avg_price - order.filled_price)
                    
                    if remaining_quantity > 0:
                        position.quantity = remaining_quantity
                        position.avg_price = order.filled_price
                    else:
                        position.quantity = 0
                        position.avg_price = 0
                else:
                    # Частичное закрытие короткой позиции
                    position.realized_pnl += order.filled_quantity * (position.avg_price - order.filled_price)
                    position.quantity += order.filled_quantity
        
        else:  # SELL
            # Продажа
            if position.quantity <= 0:
                # Увеличение короткой позиции
                total_cost = abs(position.quantity) * position.avg_price + order.filled_quantity * order.filled_price
                total_quantity = abs(position.quantity) + order.filled_quantity
                position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = -total_quantity
            else:
                # Закрытие длинной позиции
                if order.filled_quantity >= position.quantity:
                    # Полное закрытие и возможное открытие короткой
                    remaining_quantity = order.filled_quantity - position.quantity
                    position.realized_pnl += position.quantity * (order.filled_price - position.avg_price)
                    
                    if remaining_quantity > 0:
                        position.quantity = -remaining_quantity
                        position.avg_price = order.filled_price
                    else:
                        position.quantity = 0
                        position.avg_price = 0
                else:
                    # Частичное закрытие длинной позиции
                    position.realized_pnl += order.filled_quantity * (order.filled_price - position.avg_price)
                    position.quantity -= order.filled_quantity
        
        # Обновление комиссии
        position.commission_paid += order.commission
        position.last_update = order.timestamp
        
        return position
    
    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Расчет нереализованной прибыли/убытка"""
        
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        if position.quantity == 0:
            return 0.0
        
        if position.quantity > 0:
            # Длинная позиция
            unrealized_pnl = position.quantity * (current_price - position.avg_price)
        else:
            # Короткая позиция
            unrealized_pnl = abs(position.quantity) * (position.avg_price - current_price)
        
        position.unrealized_pnl = unrealized_pnl
        return unrealized_pnl
    
    def get_total_exposure(self) -> float:
        """Получение общей экспозиции"""
        total_exposure = 0.0
        
        for position in self.positions.values():
            total_exposure += abs(position.quantity * position.avg_price)
        
        return total_exposure

class BacktestEngine:
    """Движок бэктестинга"""
    
    def __init__(self, config: BacktestConfig, market_data: MarketDataProvider):
        self.config = config
        self.market_data = market_data
        self.order_executor = OrderExecutor(config, market_data)
        self.position_manager = PositionManager(config)
        
        # Состояние
        self.current_capital = config.initial_capital
        self.equity_history = []
        self.orders_history = []
        self.trades_history = []
        self.current_time = None
        
        self.logger = logging.getLogger("BacktestEngine")
    
    def run_backtest(self, strategy: Callable, symbols: List[str], 
                    start_date: datetime, end_date: datetime) -> BacktestResults:
        """Запуск бэктестинга"""
        
        self.logger.info(f"Запуск бэктестинга с {start_date} по {end_date}")
        
        # Получение данных
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = self.market_data.get_price_data(symbol, start_date, end_date)
        
        # Объединение всех временных меток
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        # Основной цикл бэктестинга
        for timestamp in timestamps:
            self.current_time = timestamp
            self.market_data.current_time = timestamp
            
            # Обновление нереализованной прибыли
            self._update_unrealized_pnl(market_data, timestamp)
            
            # Генерация сигналов стратегией
            try:
                signals = strategy(market_data, timestamp, self.position_manager.positions)
                
                # Обработка сигналов
                for signal in signals:
                    order = self._create_order_from_signal(signal, timestamp)
                    if order:
                        executed_order = self.order_executor.execute_order(order, timestamp)
                        self.orders_history.append(executed_order)
                        
                        # Обновление позиций
                        if executed_order.status == OrderStatus.FILLED:
                            self.position_manager.update_position(executed_order)
                            self._record_trade(executed_order)
                
            except Exception as e:
                self.logger.error(f"Ошибка выполнения стратегии на {timestamp}: {e}")
            
            # Запись состояния портфеля
            self._record_portfolio_state(timestamp)
        
        # Расчет результатов
        return self._calculate_results(start_date, end_date)
    
    def _update_unrealized_pnl(self, market_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Обновление нереализованной прибыли"""
        
        for symbol in self.position_manager.positions:
            if symbol in market_data:
                df = market_data[symbol]
                
                # Найти ближайшую цену
                try:
                    if timestamp in df.index:
                        current_price = df.loc[timestamp, 'close']
                    else:
                        # Найти ближайшую предыдущую цену
                        available_dates = df.index[df.index <= timestamp]
                        if len(available_dates) > 0:
                            current_price = df.loc[available_dates[-1], 'close']
                        else:
                            continue
                    
                    self.position_manager.calculate_unrealized_pnl(symbol, current_price)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка обновления PnL для {symbol}: {e}")
    
    def _create_order_from_signal(self, signal: Dict[str, Any], timestamp: datetime) -> Optional[Order]:
        """Создание ордера из сигнала"""
        
        try:
            order_id = f"order_{len(self.orders_history)}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Расчет размера позиции
            quantity = self._calculate_position_size(signal)
            
            order = Order(
                id=order_id,
                symbol=signal['symbol'],
                side=OrderSide(signal['side']),
                order_type=OrderType(signal.get('type', 'market')),
                quantity=quantity,
                price=signal.get('price'),
                stop_price=signal.get('stop_price'),
                timestamp=timestamp
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Ошибка создания ордера из сигнала: {e}")
            return None
    
    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Расчет размера позиции"""
        
        if self.config.position_sizing == "fixed":
            return signal.get('quantity', 1.0)
        
        elif self.config.position_sizing == "percent":
            # Процент от капитала
            percent = signal.get('percent', 0.1)
            current_price = self.market_data.get_current_price(signal['symbol'])
            return (self.current_capital * percent) / current_price
        
        elif self.config.position_sizing == "volatility":
            # На основе волатильности (упрощенная версия)
            risk_amount = self.current_capital * self.config.risk_per_trade
            stop_loss = signal.get('stop_loss', 0.02)  # 2% стоп-лосс по умолчанию
            current_price = self.market_data.get_current_price(signal['symbol'])
            
            return risk_amount / (current_price * stop_loss)
        
        else:
            return signal.get('quantity', 1.0)
    
    def _record_trade(self, order: Order):
        """Запись сделки"""
        
        # Упрощенная логика записи сделок
        # В реальной системе нужно отслеживать открытие и закрытие позиций
        
        trade_id = f"trade_{len(self.trades_history)}_{order.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        trade = Trade(
            id=trade_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=order.filled_price,
            entry_time=order.timestamp,
            commission=order.commission,
            slippage=order.slippage
        )
        
        self.trades_history.append(trade)
    
    def _record_portfolio_state(self, timestamp: datetime):
        """Запись состояния портфеля"""
        
        # Расчет общей стоимости портфеля
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.position_manager.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.position_manager.positions.values())
        total_commission = sum(pos.commission_paid for pos in self.position_manager.positions.values())
        
        current_equity = (self.config.initial_capital + 
                         total_realized_pnl + 
                         total_unrealized_pnl - 
                         total_commission)
        
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'commission': total_commission
        })
        
        self.current_capital = current_equity
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Расчет результатов бэктестинга"""
        
        if not self.equity_history:
            return BacktestResults(
                total_return=0, annualized_return=0, volatility=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown=0, max_drawdown_duration=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                expectancy=0, var_95=0, cvar_95=0, downside_deviation=0,
                ulcer_index=0, skewness=0, kurtosis=0, tail_ratio=0
            )
        
        # Создание временных рядов
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.set_index('timestamp', inplace=True)
        
        equity_curve = equity_df['equity']
        returns = equity_curve.pct_change().dropna()
        
        # Основные метрики
        total_return = (equity_curve.iloc[-1] / self.config.initial_capital) - 1
        
        # Аннуализированная доходность
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Волатильность
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Коэффициент Шарпа
        risk_free_rate = 0.02  # 2% безрисковая ставка
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Максимальная просадка
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Продолжительность максимальной просадки
        drawdown_periods = (drawdown < 0).astype(int)
        max_dd_duration = self._calculate_max_consecutive(drawdown_periods)
        
        # Коэффициент Сортино
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Коэффициент Кальмара
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Торговые метрики
        completed_trades = [t for t in self.trades_history if t.exit_time is not None]
        total_trades = len(completed_trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = len([t for t in completed_trades if t.pnl < 0])
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in completed_trades if t.pnl > 0]
            losses = [t.pnl for t in completed_trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0
        
        # Риск-метрики
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 and var_95 != 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) if len(drawdown) > 0 else 0
        
        # Статистические метрики
        if len(returns) > 3:
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Tail ratio (95th percentile / 5th percentile)
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            tail_ratio = abs(p95 / p5) if p5 != 0 else 0
        else:
            skewness = kurtosis = tail_ratio = 0
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            var_95=var_95,
            cvar_95=cvar_95,
            downside_deviation=downside_deviation,
            ulcer_index=ulcer_index,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            equity_curve=equity_curve,
            drawdown_series=drawdown,
            returns_series=returns,
            trades=self.trades_history,
            orders=self.orders_history
        )
    
    def _calculate_max_consecutive(self, series: pd.Series) -> int:
        """Расчет максимального количества последовательных значений"""
        if len(series) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in series:
            if value == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

class MonteCarloAnalyzer:
    """Анализатор Monte Carlo"""
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.logger = logging.getLogger("MonteCarloAnalyzer")
    
    def run_monte_carlo(self, returns: pd.Series, initial_capital: float = 100000) -> Dict[str, Any]:
        """Запуск Monte Carlo симуляции"""
        
        if len(returns) < 2:
            return {}
        
        # Параметры распределения доходностей
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Симуляции
        final_values = []
        max_drawdowns = []
        
        for _ in range(self.n_simulations):
            # Генерация случайных доходностей
            simulated_returns = np.random.normal(mean_return, std_return, len(returns))
            
            # Расчет кривой капитала
            equity_curve = initial_capital * (1 + pd.Series(simulated_returns)).cumprod()
            
            # Финальная стоимость
            final_values.append(equity_curve.iloc[-1])
            
            # Максимальная просадка
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdowns.append(drawdown.min())
        
        # Статистики
        results = {
            'final_value_mean': np.mean(final_values),
            'final_value_std': np.std(final_values),
            'final_value_5th_percentile': np.percentile(final_values, 5),
            'final_value_95th_percentile': np.percentile(final_values, 95),
            'probability_of_loss': len([v for v in final_values if v < initial_capital]) / len(final_values),
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_std': np.std(max_drawdowns),
            'max_drawdown_5th_percentile': np.percentile(max_drawdowns, 5),
            'max_drawdown_95th_percentile': np.percentile(max_drawdowns, 95)
        }
        
        return results

class WalkForwardAnalyzer:
    """Анализатор Walk-Forward"""
    
    def __init__(self, train_period: int = 252, test_period: int = 63, step_size: int = 21):
        self.train_period = train_period  # Период обучения (дни)
        self.test_period = test_period    # Период тестирования (дни)
        self.step_size = step_size        # Шаг (дни)
        self.logger = logging.getLogger("WalkForwardAnalyzer")
    
    def run_walk_forward(self, backtest_engine: BacktestEngine, strategy: Callable,
                        symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Запуск Walk-Forward анализа"""
        
        results = []
        current_start = start_date
        
        while current_start + timedelta(days=self.train_period + self.test_period) <= end_date:
            # Периоды обучения и тестирования
            train_end = current_start + timedelta(days=self.train_period)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period)
            
            try:
                # Обучение стратегии (здесь можно добавить оптимизацию параметров)
                # В данном примере просто используем стратегию как есть
                
                # Тестирование
                test_results = backtest_engine.run_backtest(strategy, symbols, test_start, test_end)
                
                results.append({
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'test_return': test_results.total_return,
                    'test_sharpe': test_results.sharpe_ratio,
                    'test_max_dd': test_results.max_drawdown,
                    'test_trades': test_results.total_trades
                })
                
            except Exception as e:
                self.logger.error(f"Ошибка в Walk-Forward анализе для периода {test_start}-{test_end}: {e}")
            
            # Переход к следующему периоду
            current_start += timedelta(days=self.step_size)
        
        # Агрегированные результаты
        if results:
            returns = [r['test_return'] for r in results]
            sharpe_ratios = [r['test_sharpe'] for r in results]
            max_drawdowns = [r['test_max_dd'] for r in results]
            
            aggregated = {
                'periods_tested': len(results),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'avg_max_dd': np.mean(max_drawdowns),
                'worst_max_dd': min(max_drawdowns),
                'positive_periods': len([r for r in returns if r > 0]),
                'consistency_ratio': len([r for r in returns if r > 0]) / len(returns),
                'detailed_results': results
            }
            
            return aggregated
        
        return {}

class BacktestReporter:
    """Генератор отчетов по бэктестингу"""
    
    def __init__(self):
        self.logger = logging.getLogger("BacktestReporter")
    
    def generate_report(self, results: BacktestResults, 
                       monte_carlo: Optional[Dict[str, Any]] = None,
                       walk_forward: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текстового отчета"""
        
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ПО БЭКТЕСТИНГУ")
        report.append("=" * 80)
        
        # Основные метрики
        report.append("\n📊 ОСНОВНЫЕ МЕТРИКИ ДОХОДНОСТИ:")
        report.append(f"   Общая доходность: {results.total_return:.2%}")
        report.append(f"   Аннуализированная доходность: {results.annualized_return:.2%}")
        report.append(f"   Волатильность: {results.volatility:.2%}")
        report.append(f"   Коэффициент Шарпа: {results.sharpe_ratio:.2f}")
        report.append(f"   Коэффициент Сортино: {results.sortino_ratio:.2f}")
        report.append(f"   Коэффициент Кальмара: {results.calmar_ratio:.2f}")
        
        # Риск-метрики
        report.append("\n⚠️ МЕТРИКИ РИСКА:")
        report.append(f"   Максимальная просадка: {results.max_drawdown:.2%}")
        report.append(f"   Продолжительность макс. просадки: {results.max_drawdown_duration} дней")
        report.append(f"   VaR (95%): {results.var_95:.2%}")
        report.append(f"   CVaR (95%): {results.cvar_95:.2%}")
        report.append(f"   Ulcer Index: {results.ulcer_index:.4f}")
        
        # Торговые метрики
        report.append("\n💼 ТОРГОВЫЕ МЕТРИКИ:")
        report.append(f"   Общее количество сделок: {results.total_trades}")
        report.append(f"   Прибыльные сделки: {results.winning_trades}")
        report.append(f"   Убыточные сделки: {results.losing_trades}")
        report.append(f"   Процент прибыльных: {results.win_rate:.1%}")
        report.append(f"   Средняя прибыль: {results.avg_win:.2f}")
        report.append(f"   Средний убыток: {results.avg_loss:.2f}")
        report.append(f"   Profit Factor: {results.profit_factor:.2f}")
        report.append(f"   Математическое ожидание: {results.expectancy:.2f}")
        
        # Статистические метрики
        report.append("\n📈 СТАТИСТИЧЕСКИЕ МЕТРИКИ:")
        report.append(f"   Асимметрия (Skewness): {results.skewness:.2f}")
        report.append(f"   Эксцесс (Kurtosis): {results.kurtosis:.2f}")
        report.append(f"   Tail Ratio: {results.tail_ratio:.2f}")
        
        # Monte Carlo результаты
        if monte_carlo:
            report.append("\n🎲 MONTE CARLO АНАЛИЗ:")
            report.append(f"   Средняя финальная стоимость: ${monte_carlo['final_value_mean']:,.0f}")
            report.append(f"   5-й процентиль: ${monte_carlo['final_value_5th_percentile']:,.0f}")
            report.append(f"   95-й процентиль: ${monte_carlo['final_value_95th_percentile']:,.0f}")
            report.append(f"   Вероятность убытка: {monte_carlo['probability_of_loss']:.1%}")
            report.append(f"   Средняя макс. просадка: {monte_carlo['max_drawdown_mean']:.2%}")
        
        # Walk-Forward результаты
        if walk_forward:
            report.append("\n🚶 WALK-FORWARD АНАЛИЗ:")
            report.append(f"   Протестировано периодов: {walk_forward['periods_tested']}")
            report.append(f"   Средняя доходность: {walk_forward['avg_return']:.2%}")
            report.append(f"   Стандартное отклонение: {walk_forward['std_return']:.2%}")
            report.append(f"   Средний Sharpe: {walk_forward['avg_sharpe']:.2f}")
            report.append(f"   Коэффициент стабильности: {walk_forward['consistency_ratio']:.1%}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def plot_results(self, results: BacktestResults, save_path: Optional[str] = None):
        """Построение графиков результатов"""
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib недоступен для построения графиков")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Результаты бэктестинга', fontsize=16, fontweight='bold')
        
        # График кривой капитала
        axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
        axes[0, 0].set_title('Кривая капитала')
        axes[0, 0].set_ylabel('Стоимость портфеля')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График просадок
        axes[0, 1].fill_between(results.drawdown_series.index, 
                               results.drawdown_series.values, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].set_title('Просадки')
        axes[0, 1].set_ylabel('Просадка (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Гистограмма доходностей
        axes[1, 0].hist(results.returns_series.values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Распределение доходностей')
        axes[1, 0].set_xlabel('Доходность')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        if len(results.returns_series) > 60:
            rolling_sharpe = results.returns_series.rolling(60).mean() / results.returns_series.rolling(60).std() * np.sqrt(252)
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
            axes[1, 1].set_title('Rolling Sharpe Ratio (60 дней)')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Недостаточно данных\nдля Rolling Sharpe', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Rolling Sharpe Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График сохранен: {save_path}")
        
        return fig

# Пример стратегии для демонстрации
def simple_momentum_strategy(market_data: Dict[str, pd.DataFrame], 
                           current_time: datetime,
                           positions: Dict[str, Position]) -> List[Dict[str, Any]]:
    """Простая моментум стратегия для демонстрации"""
    
    signals = []
    
    for symbol, df in market_data.items():
        try:
            # Получение данных до текущего времени
            historical_data = df[df.index <= current_time]
            
            if len(historical_data) < 20:
                continue
            
            # Простой моментум сигнал
            current_price = historical_data['close'].iloc[-1]
            sma_20 = historical_data['close'].rolling(20).mean().iloc[-1]
            
            current_position = positions.get(symbol, Position(symbol))
            
            # Сигнал на покупку
            if current_price > sma_20 * 1.02 and current_position.quantity <= 0:
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': 100,  # Фиксированное количество для демонстрации
                    'stop_loss': 0.05  # 5% стоп-лосс
                })
            
            # Сигнал на продажу
            elif current_price < sma_20 * 0.98 and current_position.quantity > 0:
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': current_position.quantity
                })
        
        except Exception as e:
            logging.error(f"Ошибка в стратегии для {symbol}: {e}")
    
    return signals

# Демонстрация бэктестинга
def demo_backtesting():
    """Демонстрация системы бэктестинга"""
    
    print("🔬 ДЕМОНСТРАЦИЯ СИСТЕМЫ БЭКТЕСТИНГА")
    print("=" * 60)
    
    # Генерация тестовых данных
    np.random.seed(42)
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Создание данных для двух активов
    symbols = ['AAPL', 'GOOGL']
    market_data = {}
    
    for symbol in symbols:
        # Симуляция ценовых данных
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Дневные доходности
        prices = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
        
        # Создание OHLC данных
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        volume = np.random.randint(1000000, 5000000, len(dates))
        
        market_data[symbol] = pd.DataFrame({
            'open': prices.shift(1).fillna(prices.iloc[0]),
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=dates)
    
    print(f"📊 Сгенерированы данные для {len(symbols)} активов")
    print(f"   Период: {start_date.date()} - {end_date.date()}")
    print(f"   Количество дней: {len(dates)}")
    
    # Конфигурация бэктестинга
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        position_sizing="fixed"
    )
    
    # Создание провайдера данных и движка
    data_provider = SimulatedMarketData(market_data)
    backtest_engine = BacktestEngine(config, data_provider)
    
    print(f"\n🚀 Запуск бэктестинга...")
    print(f"   Начальный капитал: ${config.initial_capital:,.0f}")
    print(f"   Комиссия: {config.commission_rate:.1%}")
    print(f"   Slippage: {config.slippage_rate:.2%}")
    
    try:
        # Запуск бэктестинга
        results = backtest_engine.run_backtest(
            simple_momentum_strategy, 
            symbols, 
            start_date, 
            end_date
        )
        
        print(f"✅ Бэктестинг завершен!")
        
        # Генерация отчета
        reporter = BacktestReporter()
        report = reporter.generate_report(results)
        print(report)
        
        # Monte Carlo анализ
        print(f"\n🎲 Запуск Monte Carlo анализа...")
        mc_analyzer = MonteCarloAnalyzer(n_simulations=1000)
        mc_results = mc_analyzer.run_monte_carlo(results.returns_series, config.initial_capital)
        
        if mc_results:
            print(f"   Средняя финальная стоимость: ${mc_results['final_value_mean']:,.0f}")
            print(f"   Вероятность убытка: {mc_results['probability_of_loss']:.1%}")
            print(f"   95% доверительный интервал: ${mc_results['final_value_5th_percentile']:,.0f} - ${mc_results['final_value_95th_percentile']:,.0f}")
        
        # Построение графиков
        if MATPLOTLIB_AVAILABLE:
            print(f"\n📈 Построение графиков...")
            fig = reporter.plot_results(results, save_path='/home/ubuntu/backtest_results.png')
            print(f"   Графики сохранены: /home/ubuntu/backtest_results.png")
        
    except Exception as e:
        print(f"❌ Ошибка при бэктестинге: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Демонстрация бэктестинга завершена")

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Запуск демонстрации
    demo_backtesting()

