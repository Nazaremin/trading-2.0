"""
Упрощенная демонстрация интегрированной торговой системы

Демонстрирует работу всех компонентов без внешних зависимостей
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class TradingSystemConfig:
    """Конфигурация торговой системы"""
    system_name: str = "Advanced Trading System Demo"
    trading_mode: str = "demo"
    forex_symbols: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    crypto_symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
    max_portfolio_risk: float = 0.02
    max_positions: int = 10

@dataclass
class TradingSignal:
    """Торговый сигнал"""
    symbol: str
    side: str
    confidence: float
    strength: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class DemoTradingSystem:
    """Демонстрационная торговая система"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.logger = logging.getLogger("DemoTradingSystem")
        self.portfolio_value = 100000.0  # $100k demo
        self.positions = {}
        self.total_pnl = 0.0
    
    def generate_market_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Генерация рыночных данных"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Симуляция цен с трендом
        np.random.seed(hash(symbol) % 2**32)
        trend = 0.0001 if 'BTC' in symbol else 0.00005
        returns = np.random.normal(trend, 0.01, periods)
        prices = 100 * (1 + pd.Series(returns)).cumprod()
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(prices.iloc[0]),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)
    
    def technical_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Технический анализ"""
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        # Сигнал на основе пересечения SMA
        if current_price > sma_20 > sma_50 and rsi < 70:
            return {"signal": 1, "confidence": 0.7, "rsi": rsi}
        elif current_price < sma_20 < sma_50 and rsi > 30:
            return {"signal": -1, "confidence": 0.7, "rsi": rsi}
        else:
            return {"signal": 0, "confidence": 0.3, "rsi": rsi}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def sentiment_analysis(self, symbol: str) -> Dict[str, float]:
        """Симуляция анализа настроений"""
        # Симуляция настроений на основе времени и символа
        time_factor = np.sin(time.time() / 3600) * 0.3  # Циклические настроения
        symbol_factor = hash(symbol) % 100 / 100 - 0.5  # Случайный фактор символа
        
        sentiment_score = time_factor + symbol_factor * 0.2
        confidence = abs(sentiment_score) + 0.3
        
        return {
            "sentiment": sentiment_score,
            "confidence": min(confidence, 1.0),
            "volume": np.random.randint(100, 1000)
        }
    
    def ml_prediction(self, df: pd.DataFrame) -> Dict[str, float]:
        """Симуляция ML предсказания"""
        # Простая модель на основе моментума и волатильности
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return {"prediction": 0, "confidence": 0.1}
        
        momentum = returns.tail(5).mean()
        volatility = returns.tail(20).std()
        
        # Простая логика: положительный моментум при низкой волатильности = покупка
        if momentum > 0 and volatility < returns.std():
            prediction = momentum / volatility
        elif momentum < 0 and volatility < returns.std():
            prediction = momentum / volatility
        else:
            prediction = 0
        
        confidence = min(abs(prediction) * 2, 0.9)
        
        return {"prediction": prediction, "confidence": confidence}
    
    def portfolio_optimization(self, symbols: List[str]) -> Dict[str, float]:
        """Симуляция портфельной оптимизации"""
        # Равновесные веса с небольшими корректировками
        base_weight = 1.0 / len(symbols)
        weights = {}
        
        for i, symbol in enumerate(symbols):
            # Небольшие корректировки на основе "риска"
            risk_adjustment = (hash(symbol) % 20 - 10) / 100
            weights[symbol] = max(0.05, base_weight + risk_adjustment)
        
        # Нормализация весов
        total_weight = sum(weights.values())
        for symbol in weights:
            weights[symbol] /= total_weight
        
        return weights
    
    def risk_management(self, signal: TradingSignal) -> bool:
        """Управление рисками"""
        # Проверка максимального количества позиций
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Проверка минимальной уверенности
        if signal.confidence < 0.4:
            return False
        
        # Проверка максимального риска
        position_risk = self.portfolio_value * 0.01  # 1% риска на позицию
        if position_risk > self.portfolio_value * self.config.max_portfolio_risk:
            return False
        
        return True
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Исполнение сделки"""
        try:
            position_size = self.portfolio_value * 0.01 * signal.confidence
            
            # Симуляция исполнения
            execution_price = 100 + np.random.normal(0, 0.1)  # Симуляция цены
            
            self.positions[signal.symbol] = {
                "side": signal.side,
                "size": position_size,
                "entry_price": execution_price,
                "timestamp": signal.timestamp
            }
            
            self.logger.info(f"✅ Исполнена сделка: {signal.symbol} {signal.side} ${position_size:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка исполнения: {e}")
            return False
    
    def calculate_pnl(self) -> float:
        """Расчет P&L"""
        total_pnl = 0
        
        for symbol, position in self.positions.items():
            # Симуляция текущей цены
            current_price = position["entry_price"] * (1 + np.random.normal(0, 0.02))
            
            if position["side"] == "buy":
                pnl = (current_price - position["entry_price"]) * position["size"] / position["entry_price"]
            else:
                pnl = (position["entry_price"] - current_price) * position["size"] / position["entry_price"]
            
            total_pnl += pnl
        
        return total_pnl
    
    async def run_demo(self):
        """Запуск демонстрации"""
        
        print("🚀 ДЕМОНСТРАЦИЯ ИНТЕГРИРОВАННОЙ ТОРГОВОЙ СИСТЕМЫ")
        print("=" * 70)
        print(f"📋 Система: {self.config.system_name}")
        print(f"💰 Начальный капитал: ${self.portfolio_value:,.2f}")
        print(f"📈 Forex символы: {self.config.forex_symbols}")
        print(f"🪙 Crypto символы: {self.config.crypto_symbols}")
        print()
        
        all_symbols = self.config.forex_symbols + self.config.crypto_symbols
        
        # Цикл торговли
        for cycle in range(3):
            print(f"🔄 ТОРГОВЫЙ ЦИКЛ {cycle + 1}")
            print("-" * 50)
            
            signals = []
            
            # Анализ каждого символа
            for symbol in all_symbols:
                print(f"\n📊 Анализ {symbol}:")
                
                # Получение данных
                market_data = self.generate_market_data(symbol)
                
                # Технический анализ
                tech_analysis = self.technical_analysis(market_data)
                print(f"   🔧 Технический: сигнал={tech_analysis['signal']}, RSI={tech_analysis['rsi']:.1f}")
                
                # Анализ настроений
                sentiment = self.sentiment_analysis(symbol)
                print(f"   😊 Настроения: {sentiment['sentiment']:.2f} (уверенность: {sentiment['confidence']:.2f})")
                
                # ML предсказание
                ml_pred = self.ml_prediction(market_data)
                print(f"   🤖 ML предсказание: {ml_pred['prediction']:.3f} (уверенность: {ml_pred['confidence']:.2f})")
                
                # Агрегация сигналов
                combined_signal = (
                    tech_analysis['signal'] * 0.4 +
                    sentiment['sentiment'] * 0.3 +
                    ml_pred['prediction'] * 0.3
                )
                
                combined_confidence = (
                    tech_analysis['confidence'] * 0.4 +
                    sentiment['confidence'] * 0.3 +
                    ml_pred['confidence'] * 0.3
                )
                
                if abs(combined_signal) > 0.2:
                    side = "buy" if combined_signal > 0 else "sell"
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        side=side,
                        confidence=combined_confidence,
                        strength=abs(combined_signal),
                        timestamp=datetime.now(),
                        source="integrated",
                        metadata={
                            "technical": tech_analysis['signal'],
                            "sentiment": sentiment['sentiment'],
                            "ml_prediction": ml_pred['prediction']
                        }
                    )
                    
                    signals.append(signal)
                    print(f"   🎯 СИГНАЛ: {side.upper()} (уверенность: {combined_confidence:.2f})")
            
            # Портфельная оптимизация
            if signals:
                print(f"\n📊 ПОРТФЕЛЬНАЯ ОПТИМИЗАЦИЯ:")
                signal_symbols = [s.symbol for s in signals]
                optimal_weights = self.portfolio_optimization(signal_symbols)
                
                for symbol, weight in optimal_weights.items():
                    print(f"   {symbol}: {weight:.1%}")
            
            # Управление рисками и исполнение
            print(f"\n⚠️ УПРАВЛЕНИЕ РИСКАМИ И ИСПОЛНЕНИЕ:")
            executed_trades = 0
            
            for signal in signals:
                if self.risk_management(signal):
                    if self.execute_trade(signal):
                        executed_trades += 1
                else:
                    print(f"   ❌ Сигнал {signal.symbol} отклонен риск-менеджментом")
            
            print(f"   ✅ Исполнено сделок: {executed_trades}/{len(signals)}")
            
            # Обновление P&L
            current_pnl = self.calculate_pnl()
            self.total_pnl += current_pnl
            
            print(f"\n💰 РЕЗУЛЬТАТЫ ЦИКЛА:")
            print(f"   Активные позиции: {len(self.positions)}")
            print(f"   P&L цикла: ${current_pnl:,.2f}")
            print(f"   Общая P&L: ${self.total_pnl:,.2f}")
            print(f"   Стоимость портфеля: ${self.portfolio_value + self.total_pnl:,.2f}")
            
            # Пауза между циклами
            if cycle < 2:
                print(f"\n⏳ Пауза 3 секунды...")
                await asyncio.sleep(3)
        
        # Итоговые результаты
        print(f"\n🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 70)
        print(f"💰 Начальный капитал: ${self.portfolio_value:,.2f}")
        print(f"📈 Итоговая стоимость: ${self.portfolio_value + self.total_pnl:,.2f}")
        print(f"💵 Общая прибыль/убыток: ${self.total_pnl:,.2f}")
        print(f"📊 Доходность: {(self.total_pnl / self.portfolio_value) * 100:.2f}%")
        print(f"🎯 Всего позиций: {len(self.positions)}")
        
        if self.positions:
            print(f"\n📋 АКТИВНЫЕ ПОЗИЦИИ:")
            for symbol, position in self.positions.items():
                print(f"   {symbol}: {position['side']} ${position['size']:.2f} @ ${position['entry_price']:.2f}")
        
        print(f"\n✅ Демонстрация завершена!")

async def main():
    """Главная функция демонстрации"""
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Конфигурация системы
    config = TradingSystemConfig(
        system_name="Advanced AI Trading System v2.0",
        forex_symbols=['EURUSD', 'GBPUSD'],
        crypto_symbols=['BTCUSDT', 'ETHUSDT'],
        max_positions=6
    )
    
    # Создание и запуск системы
    trading_system = DemoTradingSystem(config)
    await trading_system.run_demo()

if __name__ == "__main__":
    asyncio.run(main())

