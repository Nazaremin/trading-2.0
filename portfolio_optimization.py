"""
Модуль портфельной оптимизации и управления корреляциями
Включает:
- Оптимизацию портфеля по Марковицу
- Black-Litterman модель
- Risk Parity подход
- Динамическое управление корреляциями
- Многофакторные модели риска
- Оптимизацию с ограничениями
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Оптимизация и математика
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy import linalg
    from scipy.stats import norm, multivariate_normal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy не установлен. Установите: pip install scipy")

# Продвинутая оптимизация портфеля
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY не установлен. Установите: pip install cvxpy")

# Статистика и эконометрика
try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.stats.correlation_tools import corr_nearest
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels не установлен. Установите: pip install statsmodels")

@dataclass
class PortfolioConfig:
    """Конфигурация портфеля"""
    optimization_method: str = "markowitz"  # markowitz, black_litterman, risk_parity, equal_weight
    risk_model: str = "sample"  # sample, factor, shrinkage
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    max_weight: float = 0.3  # Максимальный вес актива
    min_weight: float = 0.0  # Минимальный вес актива
    target_return: Optional[float] = None
    risk_aversion: float = 1.0
    transaction_costs: float = 0.001  # 0.1%
    lookback_period: int = 252  # дней для расчета статистик
    confidence_level: float = 0.95

@dataclass
class Asset:
    """Информация об активе"""
    symbol: str
    name: str
    asset_class: str  # equity, bond, commodity, currency, crypto
    sector: Optional[str] = None
    country: Optional[str] = None
    market_cap: Optional[float] = None
    liquidity_score: float = 1.0
    esg_score: Optional[float] = None

@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float
    weights: Dict[str, float]
    correlations: pd.DataFrame

@dataclass
class RiskFactors:
    """Факторы риска"""
    market_factor: float = 0.0
    size_factor: float = 0.0
    value_factor: float = 0.0
    momentum_factor: float = 0.0
    quality_factor: float = 0.0
    volatility_factor: float = 0.0
    currency_factor: float = 0.0
    sector_factors: Dict[str, float] = field(default_factory=dict)

class RiskModel(ABC):
    """Абстрактная модель риска"""
    
    @abstractmethod
    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Оценка ковариационной матрицы"""
        pass
    
    @abstractmethod
    def estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Оценка ожидаемых доходностей"""
        pass

class SampleRiskModel(RiskModel):
    """Выборочная модель риска"""
    
    def __init__(self, shrinkage_factor: float = 0.1):
        self.shrinkage_factor = shrinkage_factor
        self.logger = logging.getLogger("SampleRiskModel")
    
    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Оценка ковариационной матрицы с shrinkage"""
        sample_cov = returns.cov().values
        
        # Shrinkage к диагональной матрице
        n_assets = sample_cov.shape[0]
        target = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        
        shrunk_cov = (1 - self.shrinkage_factor) * sample_cov + self.shrinkage_factor * target
        
        return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)
    
    def estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Оценка ожидаемых доходностей"""
        # Простое среднее с корректировкой на волатильность
        mean_returns = returns.mean()
        volatilities = returns.std()
        
        # Корректировка на основе Sharpe ratio
        sharpe_ratios = mean_returns / volatilities
        median_sharpe = sharpe_ratios.median()
        
        # Shrinkage к медианному Sharpe ratio
        adjusted_returns = volatilities * median_sharpe * 0.5 + mean_returns * 0.5
        
        return adjusted_returns

class FactorRiskModel(RiskModel):
    """Факторная модель риска"""
    
    def __init__(self, factors: List[str] = None):
        if factors is None:
            factors = ['market', 'size', 'value', 'momentum']
        self.factors = factors
        self.factor_loadings = {}
        self.factor_returns = None
        self.specific_risks = None
        self.logger = logging.getLogger("FactorRiskModel")
    
    def fit_factor_model(self, returns: pd.DataFrame, factor_returns: pd.DataFrame):
        """Подгонка факторной модели"""
        self.factor_returns = factor_returns
        self.factor_loadings = {}
        self.specific_risks = {}
        
        for asset in returns.columns:
            # Регрессия доходностей актива на факторы
            y = returns[asset].dropna()
            X = factor_returns.loc[y.index]
            
            # Добавление константы
            X = X.copy()
            X['const'] = 1.0
            
            try:
                # OLS регрессия
                beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
                
                # Сохранение факторных нагрузок
                self.factor_loadings[asset] = dict(zip(X.columns, beta))
                
                # Специфический риск
                predicted = X.values @ beta
                residuals = y.values - predicted
                self.specific_risks[asset] = np.var(residuals)
                
            except Exception as e:
                self.logger.error(f"Ошибка подгонки модели для {asset}: {e}")
                # Fallback значения
                self.factor_loadings[asset] = {factor: 0.0 for factor in self.factors}
                self.factor_loadings[asset]['const'] = 0.0
                self.specific_risks[asset] = returns[asset].var()
    
    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Оценка ковариационной матрицы через факторную модель"""
        if not self.factor_loadings:
            # Fallback к выборочной ковариации
            return returns.cov()
        
        assets = returns.columns
        n_assets = len(assets)
        
        # Матрица факторных нагрузок
        B = np.zeros((n_assets, len(self.factors)))
        for i, asset in enumerate(assets):
            for j, factor in enumerate(self.factors):
                B[i, j] = self.factor_loadings.get(asset, {}).get(factor, 0.0)
        
        # Ковариационная матрица факторов
        if self.factor_returns is not None:
            F = self.factor_returns[self.factors].cov().values
        else:
            F = np.eye(len(self.factors)) * 0.01  # Fallback
        
        # Диагональная матрица специфических рисков
        D = np.diag([self.specific_risks.get(asset, 0.01) for asset in assets])
        
        # Ковариационная матрица: B * F * B' + D
        cov_matrix = B @ F @ B.T + D
        
        return pd.DataFrame(cov_matrix, index=assets, columns=assets)
    
    def estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Оценка ожидаемых доходностей через факторную модель"""
        if not self.factor_loadings or self.factor_returns is None:
            return returns.mean()
        
        # Ожидаемые доходности факторов
        factor_expected_returns = self.factor_returns[self.factors].mean()
        
        expected_returns = {}
        for asset in returns.columns:
            loadings = self.factor_loadings.get(asset, {})
            expected_return = sum(
                loadings.get(factor, 0.0) * factor_expected_returns.get(factor, 0.0)
                for factor in self.factors
            )
            expected_returns[asset] = expected_return
        
        return pd.Series(expected_returns)

class PortfolioOptimizer:
    """Оптимизатор портфеля"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = logging.getLogger("PortfolioOptimizer")
        
        # Выбор модели риска
        if config.risk_model == "factor":
            self.risk_model = FactorRiskModel()
        else:
            self.risk_model = SampleRiskModel()
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          market_views: Optional[Dict[str, float]] = None,
                          benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Оптимизация портфеля"""
        
        if self.config.optimization_method == "markowitz":
            return self._optimize_markowitz(returns)
        elif self.config.optimization_method == "black_litterman":
            return self._optimize_black_litterman(returns, market_views, benchmark_weights)
        elif self.config.optimization_method == "risk_parity":
            return self._optimize_risk_parity(returns)
        elif self.config.optimization_method == "equal_weight":
            return self._optimize_equal_weight(returns)
        else:
            raise ValueError(f"Неизвестный метод оптимизации: {self.config.optimization_method}")
    
    def _optimize_markowitz(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Оптимизация по Марковицу"""
        # Оценка параметров
        expected_returns = self.risk_model.estimate_expected_returns(returns)
        cov_matrix = self.risk_model.estimate_covariance(returns)
        
        n_assets = len(returns.columns)
        
        # Целевая функция: максимизация utility = return - 0.5 * risk_aversion * risk
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -(portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk ** 2)
        
        # Ограничения
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Сумма весов = 1
        ]
        
        # Добавление целевой доходности если указана
        if self.config.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - self.config.target_return
            })
        
        # Границы весов
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Начальное приближение
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Оптимизация
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = dict(zip(returns.columns, result.x))
                return {k: max(0, v) for k, v in weights.items()}  # Убираем отрицательные веса
            else:
                self.logger.warning("Оптимизация не сошлась, используем равные веса")
                return self._optimize_equal_weight(returns)
                
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации Марковица: {e}")
            return self._optimize_equal_weight(returns)
    
    def _optimize_black_litterman(self, returns: pd.DataFrame,
                                 market_views: Optional[Dict[str, float]] = None,
                                 benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Оптимизация Black-Litterman"""
        
        # Если нет рыночных весов, используем равные
        if benchmark_weights is None:
            benchmark_weights = {asset: 1.0/len(returns.columns) for asset in returns.columns}
        
        # Ковариационная матрица
        cov_matrix = self.risk_model.estimate_covariance(returns).values
        
        # Рыночные веса
        w_market = np.array([benchmark_weights.get(asset, 0) for asset in returns.columns])
        
        # Подразумеваемые ожидаемые доходности (reverse optimization)
        risk_aversion = self.config.risk_aversion
        implied_returns = risk_aversion * np.dot(cov_matrix, w_market)
        
        # Если есть взгляды аналитика
        if market_views:
            # Матрица P (какие активы затрагивают взгляды)
            views_assets = list(market_views.keys())
            P = np.zeros((len(views_assets), len(returns.columns)))
            Q = np.zeros(len(views_assets))  # Вектор взглядов
            
            for i, asset in enumerate(views_assets):
                if asset in returns.columns:
                    asset_idx = list(returns.columns).index(asset)
                    P[i, asset_idx] = 1.0
                    Q[i] = market_views[asset]
            
            # Матрица неопределенности взглядов (Omega)
            Omega = np.eye(len(views_assets)) * 0.01  # 1% неопределенность
            
            # Параметр tau (неопределенность prior)
            tau = 1.0 / len(returns)
            
            # Black-Litterman формула
            try:
                M1 = linalg.inv(tau * cov_matrix)
                M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
                M3 = np.dot(linalg.inv(tau * cov_matrix), implied_returns)
                M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
                
                # Новые ожидаемые доходности
                mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
                
                # Новая ковариационная матрица
                cov_bl = linalg.inv(M1 + M2)
                
            except Exception as e:
                self.logger.error(f"Ошибка в Black-Litterman: {e}")
                mu_bl = implied_returns
                cov_bl = cov_matrix
        else:
            mu_bl = implied_returns
            cov_bl = cov_matrix
        
        # Оптимизация с новыми параметрами
        def objective(weights):
            portfolio_return = np.dot(weights, mu_bl)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_bl, weights)))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_risk ** 2)
        
        # Ограничения и границы
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(len(returns.columns))]
        x0 = w_market
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(returns.columns, result.x))
            else:
                return dict(zip(returns.columns, w_market))
                
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации Black-Litterman: {e}")
            return dict(zip(returns.columns, w_market))
    
    def _optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Оптимизация Risk Parity"""
        cov_matrix = self.risk_model.estimate_covariance(returns).values
        n_assets = len(returns.columns)
        
        def risk_budget_objective(weights):
            """Целевая функция для Risk Parity"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Маргинальный вклад в риск
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Вклад в риск
            contrib = weights * marginal_contrib
            
            # Целевой вклад (равный для всех активов)
            target_contrib = portfolio_vol / n_assets
            
            # Минимизируем отклонение от равного вклада
            return np.sum((contrib - target_contrib) ** 2)
        
        # Ограничения
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Начальное приближение - обратно пропорционально волатильности
        initial_weights = 1.0 / np.diag(cov_matrix)
        initial_weights = initial_weights / np.sum(initial_weights)
        
        try:
            result = minimize(risk_budget_objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(returns.columns, result.x))
            else:
                return self._optimize_equal_weight(returns)
                
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации Risk Parity: {e}")
            return self._optimize_equal_weight(returns)
    
    def _optimize_equal_weight(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Равновзвешенный портфель"""
        n_assets = len(returns.columns)
        equal_weight = 1.0 / n_assets
        return {asset: equal_weight for asset in returns.columns}

class CorrelationManager:
    """Менеджер корреляций"""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.logger = logging.getLogger("CorrelationManager")
    
    def calculate_dynamic_correlations(self, returns: pd.DataFrame, 
                                     method: str = "ewm") -> pd.DataFrame:
        """Расчет динамических корреляций"""
        
        if method == "ewm":
            # Экспоненциально взвешенные корреляции
            return returns.ewm(span=self.lookback_window).corr().iloc[-len(returns.columns):]
        
        elif method == "rolling":
            # Скользящие корреляции
            return returns.rolling(window=self.lookback_window).corr().iloc[-len(returns.columns):]
        
        elif method == "dcc":
            # Dynamic Conditional Correlation (упрощенная версия)
            return self._calculate_dcc(returns)
        
        else:
            # Статические корреляции
            return returns.corr()
    
    def _calculate_dcc(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Упрощенная DCC модель"""
        # Стандартизация доходностей
        standardized_returns = returns / returns.rolling(window=22).std()
        
        # Экспоненциально взвешенные корреляции
        ewm_corr = standardized_returns.ewm(span=60).corr()
        
        return ewm_corr.iloc[-len(returns.columns):]
    
    def detect_correlation_regimes(self, returns: pd.DataFrame, 
                                  window: int = 60) -> pd.Series:
        """Детекция режимов корреляций"""
        
        # Скользящие средние корреляции
        rolling_corr = []
        
        for i in range(window, len(returns)):
            subset = returns.iloc[i-window:i]
            avg_corr = subset.corr().values[np.triu_indices_from(subset.corr().values, k=1)].mean()
            rolling_corr.append(avg_corr)
        
        corr_series = pd.Series(rolling_corr, index=returns.index[window:])
        
        # Определение режимов на основе квантилей
        low_threshold = corr_series.quantile(0.33)
        high_threshold = corr_series.quantile(0.67)
        
        regimes = pd.Series(index=corr_series.index, dtype=str)
        regimes[corr_series <= low_threshold] = "low_correlation"
        regimes[(corr_series > low_threshold) & (corr_series <= high_threshold)] = "medium_correlation"
        regimes[corr_series > high_threshold] = "high_correlation"
        
        return regimes
    
    def calculate_correlation_risk(self, weights: Dict[str, float], 
                                  correlations: pd.DataFrame) -> float:
        """Расчет риска корреляций"""
        
        assets = list(weights.keys())
        weight_vector = np.array([weights[asset] for asset in assets])
        
        # Средняя корреляция портфеля
        corr_matrix = correlations.loc[assets, assets].values
        
        # Убираем диагональ (корреляция с самим собой)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_correlation = np.average(corr_matrix[mask], weights=np.outer(weight_vector, weight_vector)[mask])
        
        return avg_correlation

class PortfolioAnalyzer:
    """Анализатор портфеля"""
    
    def __init__(self):
        self.logger = logging.getLogger("PortfolioAnalyzer")
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, 
                                   weights: Dict[str, float],
                                   benchmark_returns: Optional[pd.Series] = None,
                                   risk_free_rate: float = 0.02) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        
        # Доходности портфеля
        portfolio_returns = self._calculate_portfolio_returns(returns, weights)
        
        # Базовые метрики
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Максимальная просадка
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk и Conditional VaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Метрики относительно бенчмарка
        if benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            # Beta и Alpha
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = (expected_return - risk_free_rate) - beta * (benchmark_returns.mean() * 252 - risk_free_rate)
        else:
            tracking_error = 0
            information_ratio = 0
            beta = 0
            alpha = 0
        
        # Корреляционная матрица
        correlations = returns[list(weights.keys())].corr()
        
        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            beta=beta,
            alpha=alpha,
            weights=weights,
            correlations=correlations
        )
    
    def _calculate_portfolio_returns(self, returns: pd.DataFrame, 
                                   weights: Dict[str, float]) -> pd.Series:
        """Расчет доходностей портфеля"""
        portfolio_returns = pd.Series(0, index=returns.index)
        
        for asset, weight in weights.items():
            if asset in returns.columns:
                portfolio_returns += weight * returns[asset]
        
        return portfolio_returns
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame, 
                                   n_portfolios: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Расчет эффективной границы"""
        
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Диапазон целевых доходностей
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                # Оптимизация для каждой целевой доходности
                n_assets = len(returns.columns)
                
                def objective(weights):
                    return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_ret}
                ]
                
                bounds = [(0, 1) for _ in range(n_assets)]
                x0 = np.array([1.0 / n_assets] * n_assets)
                
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    portfolio_return = np.dot(result.x, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                    efficient_portfolios.append((portfolio_risk, portfolio_return))
                
            except Exception as e:
                self.logger.error(f"Ошибка расчета эффективной границы для доходности {target_ret}: {e}")
        
        if efficient_portfolios:
            risks, returns_eff = zip(*efficient_portfolios)
            return np.array(risks), np.array(returns_eff)
        else:
            return np.array([]), np.array([])

# Пример использования
def demo_portfolio_optimization():
    """Демонстрация портфельной оптимизации"""
    
    print("📊 ДЕМОНСТРАЦИЯ ПОРТФЕЛЬНОЙ ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    # Генерация примерных данных
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Симуляция доходностей для 5 активов
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    n_assets = len(assets)
    
    # Корреляционная матрица
    correlation_matrix = np.array([
        [1.00, 0.60, 0.65, 0.30, 0.55],
        [0.60, 1.00, 0.70, 0.25, 0.60],
        [0.65, 0.70, 1.00, 0.20, 0.50],
        [0.30, 0.25, 0.20, 1.00, 0.35],
        [0.55, 0.60, 0.50, 0.35, 1.00]
    ])
    
    # Волатильности
    volatilities = np.array([0.25, 0.28, 0.22, 0.45, 0.30])
    
    # Ковариационная матрица
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Генерация доходностей
    mean_returns = np.array([0.12, 0.10, 0.11, 0.15, 0.13]) / 252  # Дневные доходности
    
    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix / 252, len(dates))
    returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
    
    print(f"📈 Сгенерированы данные для {len(assets)} активов за {len(dates)} дней")
    print(f"   Активы: {', '.join(assets)}")
    
    # Конфигурация портфеля
    config = PortfolioConfig(
        optimization_method="markowitz",
        max_weight=0.4,
        min_weight=0.05,
        risk_aversion=2.0
    )
    
    # Создание оптимизатора
    optimizer = PortfolioOptimizer(config)
    
    print(f"\n🔧 Оптимизация портфеля методом: {config.optimization_method}")
    
    try:
        # Оптимизация портфеля
        optimal_weights = optimizer.optimize_portfolio(returns_df)
        
        print(f"✅ Оптимальные веса портфеля:")
        for asset, weight in optimal_weights.items():
            print(f"   {asset}: {weight:.1%}")
        
        # Анализ портфеля
        analyzer = PortfolioAnalyzer()
        metrics = analyzer.calculate_portfolio_metrics(returns_df, optimal_weights)
        
        print(f"\n📊 Метрики портфеля:")
        print(f"   Ожидаемая доходность: {metrics.expected_return:.1%}")
        print(f"   Волатильность: {metrics.volatility:.1%}")
        print(f"   Коэффициент Шарпа: {metrics.sharpe_ratio:.2f}")
        print(f"   Максимальная просадка: {metrics.max_drawdown:.1%}")
        print(f"   VaR (95%): {metrics.var_95:.1%}")
        print(f"   Коэффициент Кальмара: {metrics.calmar_ratio:.2f}")
        
        # Сравнение с равновзвешенным портфелем
        equal_weights = {asset: 1.0/len(assets) for asset in assets}
        equal_metrics = analyzer.calculate_portfolio_metrics(returns_df, equal_weights)
        
        print(f"\n⚖️ Сравнение с равновзвешенным портфелем:")
        print(f"   Улучшение Sharpe ratio: {(metrics.sharpe_ratio - equal_metrics.sharpe_ratio):.2f}")
        print(f"   Снижение волатильности: {(equal_metrics.volatility - metrics.volatility):.1%}")
        
        # Анализ корреляций
        corr_manager = CorrelationManager()
        correlations = corr_manager.calculate_dynamic_correlations(returns_df, method="ewm")
        
        print(f"\n🔗 Анализ корреляций:")
        avg_correlation = correlations.values[np.triu_indices_from(correlations.values, k=1)].mean()
        print(f"   Средняя корреляция: {avg_correlation:.2f}")
        
        correlation_risk = corr_manager.calculate_correlation_risk(optimal_weights, correlations)
        print(f"   Риск корреляций портфеля: {correlation_risk:.2f}")
        
        # Тестирование других методов оптимизации
        print(f"\n🧪 Тестирование других методов:")
        
        methods = ["risk_parity", "equal_weight"]
        for method in methods:
            config.optimization_method = method
            test_optimizer = PortfolioOptimizer(config)
            test_weights = test_optimizer.optimize_portfolio(returns_df)
            test_metrics = analyzer.calculate_portfolio_metrics(returns_df, test_weights)
            
            print(f"   {method.replace('_', ' ').title()}:")
            print(f"     Sharpe ratio: {test_metrics.sharpe_ratio:.2f}")
            print(f"     Волатильность: {test_metrics.volatility:.1%}")
        
    except Exception as e:
        print(f"❌ Ошибка при оптимизации портфеля: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Демонстрация портфельной оптимизации завершена")

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Запуск демонстрации
    demo_portfolio_optimization()

