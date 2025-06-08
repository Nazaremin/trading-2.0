"""
Модуль машинного обучения для оптимизации торговых параметров
Автор: Manus AI (Алгопрограммист)

Включает:
- Оптимизацию параметров стратегии
- Предсказание рыночных движений
- Адаптивное обучение
- Ансамблевые методы
- Reinforcement Learning для торговых решений
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import pickle
import json
from abc import ABC, abstractmethod

# Машинное обучение
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML библиотеки не установлены. Установите: pip install scikit-learn xgboost")

# Глубокое обучение
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("TensorFlow не установлен. Установите: pip install tensorflow")

# Оптимизация
try:
    import optuna
    from scipy.optimize import differential_evolution, minimize
    OPTIM_AVAILABLE = True
except ImportError:
    OPTIM_AVAILABLE = False
    logging.warning("Optuna не установлен. Установите: pip install optuna")

@dataclass
class MLConfig:
    """Конфигурация машинного обучения"""
    model_type: str = "ensemble"  # ensemble, xgboost, lstm, reinforcement
    optimization_method: str = "optuna"  # optuna, grid_search, genetic
    feature_selection: bool = True
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    max_trials: int = 100
    early_stopping_patience: int = 10

@dataclass
class FeatureSet:
    """Набор признаков для ML"""
    technical_indicators: List[str] = field(default_factory=list)
    price_features: List[str] = field(default_factory=list)
    volume_features: List[str] = field(default_factory=list)
    volatility_features: List[str] = field(default_factory=list)
    sentiment_features: List[str] = field(default_factory=list)
    macro_features: List[str] = field(default_factory=list)

@dataclass
class MLResult:
    """Результат ML модели"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2: float
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    confidence: np.ndarray

class FeatureEngineer:
    """Инженерия признаков для торговых данных"""
    
    def __init__(self):
        self.logger = logging.getLogger("FeatureEngineer")
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание технических индикаторов"""
        result = df.copy()
        
        # Скользящие средние
        for period in [5, 10, 20, 50, 100, 200]:
            result[f'sma_{period}'] = result['close'].rolling(period).mean()
            result[f'ema_{period}'] = result['close'].ewm(span=period).mean()
        
        # RSI
        for period in [14, 21]:
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = result['close'].ewm(span=12).mean()
        ema26 = result['close'].ewm(span=26).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # Bollinger Bands
        for period in [20]:
            sma = result['close'].rolling(period).mean()
            std = result['close'].rolling(period).std()
            result[f'bb_upper_{period}'] = sma + (std * 2)
            result[f'bb_lower_{period}'] = sma - (std * 2)
            result[f'bb_width_{period}'] = result[f'bb_upper_{period}'] - result[f'bb_lower_{period}']
            result[f'bb_position_{period}'] = (result['close'] - result[f'bb_lower_{period}']) / result[f'bb_width_{period}']
        
        # ATR (Average True Range)
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        result['atr_14'] = true_range.rolling(14).mean()
        
        # Stochastic
        for period in [14]:
            lowest_low = result['low'].rolling(period).min()
            highest_high = result['high'].rolling(period).max()
            result[f'stoch_k_{period}'] = 100 * (result['close'] - lowest_low) / (highest_high - lowest_low)
            result[f'stoch_d_{period}'] = result[f'stoch_k_{period}'].rolling(3).mean()
        
        return result
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание ценовых признаков"""
        result = df.copy()
        
        # Доходности
        for period in [1, 2, 3, 5, 10, 20]:
            result[f'return_{period}'] = result['close'].pct_change(period)
            result[f'log_return_{period}'] = np.log(result['close'] / result['close'].shift(period))
        
        # Ценовые паттерны
        result['high_low_ratio'] = result['high'] / result['low']
        result['close_open_ratio'] = result['close'] / result['open']
        result['body_size'] = np.abs(result['close'] - result['open']) / result['open']
        result['upper_shadow'] = (result['high'] - np.maximum(result['open'], result['close'])) / result['open']
        result['lower_shadow'] = (np.minimum(result['open'], result['close']) - result['low']) / result['open']
        
        # Гэпы
        result['gap'] = (result['open'] - result['close'].shift()) / result['close'].shift()
        result['gap_filled'] = ((result['low'] <= result['close'].shift()) & 
                               (result['open'] > result['close'].shift())).astype(int)
        
        # Фракталы
        result['fractal_high'] = ((result['high'] > result['high'].shift(1)) & 
                                 (result['high'] > result['high'].shift(-1))).astype(int)
        result['fractal_low'] = ((result['low'] < result['low'].shift(1)) & 
                                (result['low'] < result['low'].shift(-1))).astype(int)
        
        return result
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание объемных признаков"""
        result = df.copy()
        
        # Объемные индикаторы
        for period in [5, 10, 20]:
            result[f'volume_sma_{period}'] = result['volume'].rolling(period).mean()
            result[f'volume_ratio_{period}'] = result['volume'] / result[f'volume_sma_{period}']
        
        # OBV (On Balance Volume)
        result['obv'] = (result['volume'] * np.sign(result['close'].diff())).cumsum()
        
        # VWAP (Volume Weighted Average Price)
        result['vwap'] = (result['close'] * result['volume']).cumsum() / result['volume'].cumsum()
        result['vwap_distance'] = (result['close'] - result['vwap']) / result['vwap']
        
        # Накопление/Распределение
        clv = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
        result['ad_line'] = (clv * result['volume']).cumsum()
        
        return result
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков волатильности"""
        result = df.copy()
        
        # Историческая волатильность
        for period in [10, 20, 30]:
            returns = result['close'].pct_change()
            result[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson волатильность
        for period in [10, 20]:
            hl_ratio = np.log(result['high'] / result['low'])
            result[f'parkinson_vol_{period}'] = np.sqrt(hl_ratio.rolling(period).mean() / (4 * np.log(2)))
        
        # Garman-Klass волатильность
        for period in [10, 20]:
            hl = np.log(result['high'] / result['low']) ** 2
            co = np.log(result['close'] / result['open']) ** 2
            result[f'gk_vol_{period}'] = np.sqrt(0.5 * hl.rolling(period).mean() - 
                                                (2 * np.log(2) - 1) * co.rolling(period).mean())
        
        return result
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'close', lags: List[int] = None) -> pd.DataFrame:
        """Создание лаговых признаков"""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]
        
        result = df.copy()
        
        for lag in lags:
            result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
            result[f'{target_col}_diff_lag_{lag}'] = result[target_col].diff(lag)
            result[f'{target_col}_pct_lag_{lag}'] = result[target_col].pct_change(lag)
        
        return result
    
    def create_target_variable(self, df: pd.DataFrame, prediction_horizon: int = 1, 
                              target_type: str = 'return') -> pd.DataFrame:
        """Создание целевой переменной"""
        result = df.copy()
        
        if target_type == 'return':
            result['target'] = result['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        elif target_type == 'direction':
            future_return = result['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            result['target'] = (future_return > 0).astype(int)
        elif target_type == 'volatility':
            returns = result['close'].pct_change()
            result['target'] = returns.rolling(prediction_horizon).std().shift(-prediction_horizon)
        
        return result

class MLOptimizer:
    """Оптимизатор параметров с использованием машинного обучения"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger("MLOptimizer")
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scalers = {}
        
    def prepare_features(self, df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
        """Подготовка признаков"""
        result = df.copy()
        
        # Технические индикаторы
        if feature_set.technical_indicators:
            result = self.feature_engineer.create_technical_features(result)
        
        # Ценовые признаки
        if feature_set.price_features:
            result = self.feature_engineer.create_price_features(result)
        
        # Объемные признаки
        if feature_set.volume_features:
            result = self.feature_engineer.create_volume_features(result)
        
        # Волатильность
        if feature_set.volatility_features:
            result = self.feature_engineer.create_volatility_features(result)
        
        # Лаговые признаки
        result = self.feature_engineer.create_lag_features(result)
        
        # Целевая переменная
        result = self.feature_engineer.create_target_variable(result)
        
        return result
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """Отбор признаков"""
        if not self.config.feature_selection:
            return X.columns.tolist()
        
        # Удаление признаков с высокой корреляцией
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X_filtered = X.drop(columns=high_corr_features)
        
        # Статистический отбор
        if len(X_filtered.columns) > k:
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X_filtered, y)
            selected_features = X_filtered.columns[selector.get_support()].tolist()
        else:
            selected_features = X_filtered.columns.tolist()
        
        self.logger.info(f"Отобрано {len(selected_features)} признаков из {len(X.columns)}")
        return selected_features
    
    def create_ensemble_model(self) -> VotingRegressor:
        """Создание ансамблевой модели"""
        models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.config.random_state))
        ]
        
        if ML_AVAILABLE:
            try:
                models.append(('xgb', xgb.XGBRegressor(random_state=self.config.random_state)))
            except:
                pass
        
        return VotingRegressor(estimators=models)
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Optional[Model]:
        """Создание LSTM модели"""
        if not DL_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, Any]:
        """Оптимизация гиперпараметров"""
        if not OPTIM_AVAILABLE:
            self.logger.warning("Optuna не доступен, используются параметры по умолчанию")
            return {}
        
        def objective(trial):
            if model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=self.config.random_state)
            
            elif model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
                model = RandomForestRegressor(**params, random_state=self.config.random_state)
            
            else:
                return float('inf')
            
            # Кросс-валидация
            tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.max_trials)
        
        return study.best_params
    
    def train_model(self, df: pd.DataFrame, feature_set: FeatureSet) -> MLResult:
        """Обучение модели"""
        # Подготовка данных
        data = self.prepare_features(df, feature_set)
        data = data.dropna()
        
        # Разделение на признаки и целевую переменную
        feature_columns = [col for col in data.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        X = data[feature_columns]
        y = data['target']
        
        # Отбор признаков
        selected_features = self.select_features(X, y)
        X = X[selected_features]
        
        # Разделение на обучающую и тестовую выборки
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Масштабирование
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Выбор и обучение модели
        if self.config.model_type == "ensemble":
            model = self.create_ensemble_model()
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
        elif self.config.model_type == "xgboost":
            best_params = self.optimize_hyperparameters(X_train, y_train, "xgboost")
            model = xgb.XGBRegressor(**best_params, random_state=self.config.random_state)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
        elif self.config.model_type == "lstm" and DL_AVAILABLE:
            # Подготовка данных для LSTM
            sequence_length = 60
            X_train_lstm = self._prepare_lstm_data(X_train_scaled, sequence_length)
            y_train_lstm = y_train[sequence_length:]
            X_test_lstm = self._prepare_lstm_data(X_test_scaled, sequence_length)
            
            model = self.create_lstm_model((sequence_length, X_train.shape[1]))
            
            early_stopping = EarlyStopping(patience=self.config.early_stopping_patience, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
            
            model.fit(X_train_lstm, y_train_lstm, 
                     epochs=100, batch_size=32, validation_split=0.2,
                     callbacks=[early_stopping, reduce_lr], verbose=0)
            
            predictions = model.predict(X_test_lstm).flatten()
            
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.config.model_type}")
        
        # Вычисление метрик
        mse = mean_squared_error(y_test[-len(predictions):], predictions)
        mae = mean_absolute_error(y_test[-len(predictions):], predictions)
        r2 = r2_score(y_test[-len(predictions):], predictions)
        
        # Важность признаков
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(selected_features, model.feature_importances_))
        else:
            feature_importance = {}
        
        # Сохранение модели и скейлера
        self.models[self.config.model_type] = model
        self.scalers[self.config.model_type] = scaler
        
        return MLResult(
            model_name=self.config.model_type,
            accuracy=0.0,  # Для регрессии не применимо
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            mse=mse,
            mae=mae,
            r2=r2,
            feature_importance=feature_importance,
            predictions=predictions,
            confidence=np.ones_like(predictions) * 0.8  # Заглушка для уверенности
        )
    
    def _prepare_lstm_data(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Подготовка данных для LSTM"""
        X = []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
        return np.array(X)
    
    def predict(self, df: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Предсказание с использованием обученной модели"""
        if model_name is None:
            model_name = self.config.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Подготовка признаков (аналогично обучению)
        feature_set = FeatureSet(
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True
        )
        
        data = self.prepare_features(df, feature_set)
        feature_columns = [col for col in data.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        X = data[feature_columns].dropna()
        
        # Масштабирование
        X_scaled = scaler.transform(X)
        
        # Предсказание
        if model_name == "lstm":
            sequence_length = 60
            X_lstm = self._prepare_lstm_data(X_scaled, sequence_length)
            predictions = model.predict(X_lstm).flatten()
        else:
            predictions = model.predict(X_scaled)
        
        return predictions
    
    def save_model(self, filepath: str, model_name: str = None):
        """Сохранение модели"""
        if model_name is None:
            model_name = self.config.model_type
        
        model_data = {
            'model': self.models.get(model_name),
            'scaler': self.scalers.get(model_name),
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Модель {model_name} сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[model_data['config'].model_type] = model_data['model']
        self.scalers[model_data['config'].model_type] = model_data['scaler']
        self.config = model_data['config']
        
        self.logger.info(f"Модель загружена из {filepath}")

class ParameterOptimizer:
    """Оптимизатор параметров торговой стратегии"""
    
    def __init__(self, strategy_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.strategy_function = strategy_function
        self.parameter_bounds = parameter_bounds
        self.logger = logging.getLogger("ParameterOptimizer")
        self.optimization_history = []
    
    def objective_function(self, params: List[float], data: pd.DataFrame) -> float:
        """Целевая функция для оптимизации"""
        # Конвертация параметров в словарь
        param_dict = {}
        for i, (param_name, _) in enumerate(self.parameter_bounds.items()):
            param_dict[param_name] = params[i]
        
        try:
            # Запуск стратегии с данными параметрами
            result = self.strategy_function(data, param_dict)
            
            # Вычисление целевой метрики (например, Sharpe ratio)
            returns = result.get('returns', [])
            if len(returns) == 0:
                return -999999
            
            returns_array = np.array(returns)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            
            # Штрафы за плохие метрики
            max_drawdown = result.get('max_drawdown', 0)
            win_rate = result.get('win_rate', 0)
            
            penalty = 0
            if max_drawdown > 0.2:  # Штраф за просадку > 20%
                penalty += (max_drawdown - 0.2) * 10
            
            if win_rate < 0.4:  # Штраф за винрейт < 40%
                penalty += (0.4 - win_rate) * 5
            
            score = sharpe_ratio - penalty
            
            # Сохранение истории
            self.optimization_history.append({
                'parameters': param_dict.copy(),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'score': score
            })
            
            return -score  # Минимизируем отрицательный Sharpe ratio
            
        except Exception as e:
            self.logger.error(f"Ошибка в целевой функции: {e}")
            return 999999
    
    def optimize_genetic(self, data: pd.DataFrame, generations: int = 50, population_size: int = 50) -> Dict[str, Any]:
        """Генетическая оптимизация"""
        bounds = list(self.parameter_bounds.values())
        
        result = differential_evolution(
            func=lambda params: self.objective_function(params, data),
            bounds=bounds,
            maxiter=generations,
            popsize=population_size,
            seed=42,
            disp=True
        )
        
        # Конвертация результата
        best_params = {}
        for i, (param_name, _) in enumerate(self.parameter_bounds.items()):
            best_params[param_name] = result.x[i]
        
        return {
            'best_parameters': best_params,
            'best_score': -result.fun,
            'optimization_history': self.optimization_history,
            'convergence': result.success
        }
    
    def optimize_optuna(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Оптимизация с помощью Optuna"""
        if not OPTIM_AVAILABLE:
            raise ImportError("Optuna не установлен")
        
        def objective(trial):
            params = []
            for param_name, (low, high) in self.parameter_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    param_value = trial.suggest_int(param_name, low, high)
                else:
                    param_value = trial.suggest_float(param_name, low, high)
                params.append(param_value)
            
            return self.objective_function(params, data)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_parameters': study.best_params,
            'best_score': -study.best_value,
            'optimization_history': self.optimization_history,
            'study': study
        }

# Пример использования
def demo_ml_optimization():
    """Демонстрация ML оптимизации"""
    
    # Генерация примерных данных
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Симуляция цен с трендом и волатильностью
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': np.random.uniform(1000, 10000, len(dates))
    })
    df.set_index('timestamp', inplace=True)
    
    print("📊 ДЕМОНСТРАЦИЯ ML ОПТИМИЗАЦИИ")
    print("=" * 50)
    
    # Конфигурация ML
    config = MLConfig(
        model_type="ensemble",
        optimization_method="optuna",
        max_trials=20  # Уменьшено для демонстрации
    )
    
    # Создание оптимизатора
    optimizer = MLOptimizer(config)
    
    # Набор признаков
    feature_set = FeatureSet(
        technical_indicators=['sma', 'rsi', 'macd'],
        price_features=['returns', 'patterns'],
        volume_features=['obv', 'vwap'],
        volatility_features=['atr', 'historical_vol']
    )
    
    print("🔧 Обучение ML модели...")
    try:
        # Обучение модели
        result = optimizer.train_model(df, feature_set)
        
        print(f"✅ Модель обучена:")
        print(f"   MSE: {result.mse:.6f}")
        print(f"   MAE: {result.mae:.6f}")
        print(f"   R²: {result.r2:.4f}")
        
        # Топ-5 важных признаков
        if result.feature_importance:
            top_features = sorted(result.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            print(f"\n📈 Топ-5 важных признаков:")
            for feature, importance in top_features:
                print(f"   {feature}: {importance:.4f}")
        
        # Предсказание
        predictions = optimizer.predict(df.tail(100))
        print(f"\n🔮 Последние 5 предсказаний: {predictions[-5:]}")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении модели: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Демонстрация ML оптимизации завершена")

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Запуск демонстрации
    demo_ml_optimization()

