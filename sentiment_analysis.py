"""
Модуль анализа настроений рынка и социальных сетей
Включает:
- Анализ новостей и социальных сетей
- Интеграция с Twitter/X, Reddit, Telegram
- Обработка финансовых новостей
- Sentiment scoring и агрегация
- Корреляция с рыночными движениями
"""

import asyncio
import aiohttp
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# NLP и анализ настроений
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    import textblob
    from textblob import TextBlob
    NLTK_AVAILABLE = True
    
    # Загрузка необходимых данных NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
        
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK не установлен. Установите: pip install nltk textblob")

# Трансформеры для продвинутого анализа
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers не установлен. Установите: pip install transformers torch")

# API клиенты
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    logging.warning("Tweepy не установлен. Установите: pip install tweepy")

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    logging.warning("PRAW не установлен. Установите: pip install praw")

try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    logging.warning("Feedparser не установлен. Установите: pip install feedparser")

@dataclass
class SentimentConfig:
    """Конфигурация анализа настроений"""
    sources: List[str] = field(default_factory=lambda: ['news', 'twitter', 'reddit'])
    languages: List[str] = field(default_factory=lambda: ['en', 'ru'])
    update_interval: int = 300  # секунды
    max_posts_per_source: int = 100
    sentiment_threshold: float = 0.1
    relevance_threshold: float = 0.5
    cache_duration: int = 3600  # секунды

@dataclass
class SentimentData:
    """Данные о настроениях"""
    source: str
    timestamp: datetime
    text: str
    sentiment_score: float  # -1 до 1
    confidence: float  # 0 до 1
    relevance: float  # 0 до 1
    symbols: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedSentiment:
    """Агрегированные настроения"""
    symbol: str
    timestamp: datetime
    overall_sentiment: float
    confidence: float
    volume: int  # количество упоминаний
    sources: Dict[str, float]  # настроения по источникам
    trend: str  # 'bullish', 'bearish', 'neutral'
    strength: str  # 'weak', 'moderate', 'strong'

class SentimentAnalyzer(ABC):
    """Абстрактный базовый класс для анализа настроений"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def analyze_text(self, text: str) -> Tuple[float, float]:
        """Анализ текста, возвращает (sentiment, confidence)"""
        pass
    
    @abstractmethod
    def extract_symbols(self, text: str) -> List[str]:
        """Извлечение финансовых символов из текста"""
        pass

class VaderSentimentAnalyzer(SentimentAnalyzer):
    """Анализатор настроений на основе VADER"""
    
    def __init__(self, config: SentimentConfig):
        super().__init__(config)
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK не установлен")
        
        self.analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Финансовые ключевые слова
        self.financial_keywords = {
            'bullish': ['bull', 'bullish', 'buy', 'long', 'pump', 'moon', 'rocket', 'gain', 'profit', 'up', 'rise', 'surge'],
            'bearish': ['bear', 'bearish', 'sell', 'short', 'dump', 'crash', 'loss', 'down', 'fall', 'drop', 'decline'],
            'neutral': ['hold', 'sideways', 'flat', 'stable', 'consolidation']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        # Удаление URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Удаление упоминаний и хештегов (но сохранение символов)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(?![A-Z]{2,5}\b)', '', text)  # Удаляем хештеги, кроме тикеров
        
        # Удаление специальных символов
        text = re.sub(r'[^\w\s$#]', ' ', text)
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    async def analyze_text(self, text: str) -> Tuple[float, float]:
        """Анализ настроений текста"""
        try:
            # Предобработка
            processed_text = self.preprocess_text(text)
            
            # VADER анализ
            scores = self.analyzer.polarity_scores(processed_text)
            
            # Компаундный скор как основной показатель
            sentiment = scores['compound']
            
            # Уверенность на основе нейтральности
            confidence = 1 - scores['neu']
            
            # Дополнительная корректировка на основе финансовых ключевых слов
            financial_boost = self._calculate_financial_sentiment_boost(processed_text)
            sentiment = np.clip(sentiment + financial_boost, -1, 1)
            
            return sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа настроений: {e}")
            return 0.0, 0.0
    
    def _calculate_financial_sentiment_boost(self, text: str) -> float:
        """Дополнительная корректировка на основе финансовых терминов"""
        words = text.split()
        boost = 0.0
        
        for word in words:
            if word in self.financial_keywords['bullish']:
                boost += 0.1
            elif word in self.financial_keywords['bearish']:
                boost -= 0.1
        
        return np.clip(boost, -0.5, 0.5)
    
    def extract_symbols(self, text: str) -> List[str]:
        """Извлечение финансовых символов"""
        symbols = []
        
        # Поиск тикеров ($SYMBOL или #SYMBOL)
        ticker_pattern = r'[\$#]([A-Z]{2,5})\b'
        matches = re.findall(ticker_pattern, text.upper())
        symbols.extend(matches)
        
        # Поиск популярных криптовалют
        crypto_pattern = r'\b(BTC|ETH|ADA|DOT|LINK|UNI|DOGE|SHIB|MATIC|SOL|AVAX)\b'
        crypto_matches = re.findall(crypto_pattern, text.upper())
        symbols.extend(crypto_matches)
        
        # Поиск валютных пар
        forex_pattern = r'\b(EUR|GBP|JPY|CHF|CAD|AUD|NZD)(USD|EUR|GBP|JPY)\b'
        forex_matches = re.findall(forex_pattern, text.upper())
        symbols.extend([''.join(match) for match in forex_matches])
        
        return list(set(symbols))

class TransformerSentimentAnalyzer(SentimentAnalyzer):
    """Анализатор на основе трансформеров"""
    
    def __init__(self, config: SentimentConfig):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers не установлен")
        
        # Загрузка предобученной модели для финансовых текстов
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except:
            # Fallback на общую модель
            self.sentiment_pipeline = pipeline("sentiment-analysis")
        
        self.vader_analyzer = VaderSentimentAnalyzer(config) if NLTK_AVAILABLE else None
    
    async def analyze_text(self, text: str) -> Tuple[float, float]:
        """Анализ с использованием трансформеров"""
        try:
            # Ограничение длины текста
            if len(text) > 512:
                text = text[:512]
            
            # Анализ трансформером
            result = self.sentiment_pipeline(text)[0]
            
            # Конвертация в числовой скор
            if result['label'].upper() in ['POSITIVE', 'BULLISH']:
                sentiment = result['score']
            elif result['label'].upper() in ['NEGATIVE', 'BEARISH']:
                sentiment = -result['score']
            else:
                sentiment = 0.0
            
            confidence = result['score']
            
            # Комбинирование с VADER если доступен
            if self.vader_analyzer:
                vader_sentiment, vader_confidence = await self.vader_analyzer.analyze_text(text)
                sentiment = (sentiment + vader_sentiment) / 2
                confidence = (confidence + vader_confidence) / 2
            
            return sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа трансформером: {e}")
            # Fallback на VADER
            if self.vader_analyzer:
                return await self.vader_analyzer.analyze_text(text)
            return 0.0, 0.0
    
    def extract_symbols(self, text: str) -> List[str]:
        """Использование VADER для извлечения символов"""
        if self.vader_analyzer:
            return self.vader_analyzer.extract_symbols(text)
        return []

class NewsDataSource:
    """Источник новостных данных"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.logger = logging.getLogger("NewsDataSource")
        
        # RSS фиды финансовых новостей
        self.rss_feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reuters.com/business/finance/rss',
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.marketwatch.com/rss/topstories',
            'https://feeds.cnbc.com/cnbc/ID/100003114/device/rss/rss.html'
        ]
    
    async def fetch_news(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Получение новостей"""
        if not RSS_AVAILABLE:
            self.logger.warning("RSS парсер недоступен")
            return []
        
        news_items = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:self.config.max_posts_per_source // len(self.rss_feeds)]:
                    # Фильтрация по символам если указаны
                    if symbols:
                        text_content = f"{entry.title} {entry.get('summary', '')}"
                        found_symbols = self._extract_symbols_from_text(text_content, symbols)
                        if not found_symbols:
                            continue
                    
                    news_items.append({
                        'source': 'news',
                        'title': entry.title,
                        'content': entry.get('summary', ''),
                        'url': entry.link,
                        'published': entry.get('published_parsed'),
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                self.logger.error(f"Ошибка получения новостей из {feed_url}: {e}")
        
        return news_items
    
    def _extract_symbols_from_text(self, text: str, target_symbols: List[str]) -> List[str]:
        """Поиск целевых символов в тексте"""
        found = []
        text_upper = text.upper()
        
        for symbol in target_symbols:
            if symbol.upper() in text_upper:
                found.append(symbol)
        
        return found

class TwitterDataSource:
    """Источник данных Twitter/X"""
    
    def __init__(self, config: SentimentConfig, api_credentials: Dict[str, str]):
        self.config = config
        self.logger = logging.getLogger("TwitterDataSource")
        
        if not TWITTER_AVAILABLE:
            raise ImportError("Tweepy не установлен")
        
        # Инициализация API
        try:
            auth = tweepy.OAuthHandler(
                api_credentials['consumer_key'],
                api_credentials['consumer_secret']
            )
            auth.set_access_token(
                api_credentials['access_token'],
                api_credentials['access_token_secret']
            )
            
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Twitter API: {e}")
            self.api = None
    
    async def fetch_tweets(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Получение твитов по символам"""
        if not self.api:
            return []
        
        tweets = []
        
        for symbol in symbols:
            try:
                # Поиск твитов
                query = f"${symbol} OR #{symbol} -filter:retweets"
                search_results = tweepy.Cursor(
                    self.api.search_tweets,
                    q=query,
                    lang='en',
                    result_type='recent',
                    tweet_mode='extended'
                ).items(self.config.max_posts_per_source // len(symbols))
                
                for tweet in search_results:
                    tweets.append({
                        'source': 'twitter',
                        'id': tweet.id,
                        'text': tweet.full_text,
                        'user': tweet.user.screen_name,
                        'followers': tweet.user.followers_count,
                        'retweets': tweet.retweet_count,
                        'likes': tweet.favorite_count,
                        'timestamp': tweet.created_at,
                        'symbols': [symbol]
                    })
                    
            except Exception as e:
                self.logger.error(f"Ошибка получения твитов для {symbol}: {e}")
        
        return tweets

class RedditDataSource:
    """Источник данных Reddit"""
    
    def __init__(self, config: SentimentConfig, api_credentials: Dict[str, str]):
        self.config = config
        self.logger = logging.getLogger("RedditDataSource")
        
        if not REDDIT_AVAILABLE:
            raise ImportError("PRAW не установлен")
        
        # Инициализация Reddit API
        try:
            self.reddit = praw.Reddit(
                client_id=api_credentials['client_id'],
                client_secret=api_credentials['client_secret'],
                user_agent=api_credentials.get('user_agent', 'SentimentBot/1.0')
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Reddit API: {e}")
            self.reddit = None
        
        # Релевантные сабреддиты
        self.subreddits = [
            'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
            'cryptocurrency', 'CryptoCurrency', 'Bitcoin', 'ethereum',
            'forex', 'Forex', 'algotrading', 'SecurityAnalysis'
        ]
    
    async def fetch_posts(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Получение постов Reddit"""
        if not self.reddit:
            return []
        
        posts = []
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Получение горячих постов
                for post in subreddit.hot(limit=self.config.max_posts_per_source // len(self.subreddits)):
                    # Фильтрация по символам
                    text_content = f"{post.title} {post.selftext}"
                    found_symbols = self._extract_symbols_from_text(text_content, symbols)
                    
                    if found_symbols or not symbols:
                        posts.append({
                            'source': 'reddit',
                            'id': post.id,
                            'title': post.title,
                            'text': post.selftext,
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'timestamp': datetime.fromtimestamp(post.created_utc),
                            'symbols': found_symbols
                        })
                        
            except Exception as e:
                self.logger.error(f"Ошибка получения постов из r/{subreddit_name}: {e}")
        
        return posts
    
    def _extract_symbols_from_text(self, text: str, target_symbols: List[str]) -> List[str]:
        """Поиск символов в тексте"""
        found = []
        text_upper = text.upper()
        
        for symbol in target_symbols:
            if symbol.upper() in text_upper:
                found.append(symbol)
        
        return found

class SentimentAggregator:
    """Агрегатор настроений из различных источников"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.logger = logging.getLogger("SentimentAggregator")
        
        # Выбор анализатора настроений
        if TRANSFORMERS_AVAILABLE:
            self.sentiment_analyzer = TransformerSentimentAnalyzer(config)
        elif NLTK_AVAILABLE:
            self.sentiment_analyzer = VaderSentimentAnalyzer(config)
        else:
            raise ImportError("Ни один анализатор настроений не доступен")
        
        # Источники данных
        self.news_source = NewsDataSource(config)
        self.data_sources = {'news': self.news_source}
        
        # Кэш для хранения результатов
        self.sentiment_cache = {}
        self.last_update = {}
    
    def add_twitter_source(self, api_credentials: Dict[str, str]):
        """Добавление источника Twitter"""
        try:
            self.data_sources['twitter'] = TwitterDataSource(self.config, api_credentials)
            self.logger.info("Twitter источник добавлен")
        except Exception as e:
            self.logger.error(f"Ошибка добавления Twitter источника: {e}")
    
    def add_reddit_source(self, api_credentials: Dict[str, str]):
        """Добавление источника Reddit"""
        try:
            self.data_sources['reddit'] = RedditDataSource(self.config, api_credentials)
            self.logger.info("Reddit источник добавлен")
        except Exception as e:
            self.logger.error(f"Ошибка добавления Reddit источника: {e}")
    
    async def analyze_sentiment_for_symbols(self, symbols: List[str]) -> Dict[str, AggregatedSentiment]:
        """Анализ настроений для списка символов"""
        results = {}
        
        for symbol in symbols:
            # Проверка кэша
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
            if cache_key in self.sentiment_cache:
                results[symbol] = self.sentiment_cache[cache_key]
                continue
            
            # Сбор данных из всех источников
            all_sentiments = []
            source_sentiments = {}
            
            for source_name, source in self.data_sources.items():
                try:
                    if source_name == 'news':
                        data = await source.fetch_news([symbol])
                    elif source_name == 'twitter':
                        data = await source.fetch_tweets([symbol])
                    elif source_name == 'reddit':
                        data = await source.fetch_posts([symbol])
                    else:
                        continue
                    
                    # Анализ настроений для каждого элемента
                    source_sentiment_scores = []
                    
                    for item in data:
                        text = self._extract_text_from_item(item)
                        if text:
                            sentiment, confidence = await self.sentiment_analyzer.analyze_text(text)
                            
                            # Взвешивание по релевантности и популярности
                            weight = self._calculate_item_weight(item)
                            weighted_sentiment = sentiment * confidence * weight
                            
                            all_sentiments.append(weighted_sentiment)
                            source_sentiment_scores.append(weighted_sentiment)
                    
                    # Агрегация по источнику
                    if source_sentiment_scores:
                        source_sentiments[source_name] = np.mean(source_sentiment_scores)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка анализа {source_name} для {symbol}: {e}")
            
            # Общая агрегация
            if all_sentiments:
                overall_sentiment = np.mean(all_sentiments)
                confidence = min(1.0, len(all_sentiments) / 50)  # Больше данных = больше уверенности
                
                # Определение тренда и силы
                trend = self._determine_trend(overall_sentiment)
                strength = self._determine_strength(overall_sentiment, confidence)
                
                aggregated = AggregatedSentiment(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_sentiment=overall_sentiment,
                    confidence=confidence,
                    volume=len(all_sentiments),
                    sources=source_sentiments,
                    trend=trend,
                    strength=strength
                )
                
                # Кэширование
                self.sentiment_cache[cache_key] = aggregated
                results[symbol] = aggregated
                
            else:
                # Нейтральный результат если нет данных
                results[symbol] = AggregatedSentiment(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_sentiment=0.0,
                    confidence=0.0,
                    volume=0,
                    sources={},
                    trend='neutral',
                    strength='weak'
                )
        
        return results
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """Извлечение текста из элемента данных"""
        if item['source'] == 'news':
            return f"{item.get('title', '')} {item.get('content', '')}"
        elif item['source'] == 'twitter':
            return item.get('text', '')
        elif item['source'] == 'reddit':
            return f"{item.get('title', '')} {item.get('text', '')}"
        return ""
    
    def _calculate_item_weight(self, item: Dict[str, Any]) -> float:
        """Вычисление веса элемента"""
        weight = 1.0
        
        if item['source'] == 'twitter':
            # Вес на основе популярности твита
            followers = item.get('followers', 0)
            retweets = item.get('retweets', 0)
            likes = item.get('likes', 0)
            
            popularity_score = (followers / 10000) + (retweets / 100) + (likes / 1000)
            weight = min(3.0, 1.0 + popularity_score)
            
        elif item['source'] == 'reddit':
            # Вес на основе скора поста
            score = item.get('score', 0)
            upvote_ratio = item.get('upvote_ratio', 0.5)
            
            weight = min(2.0, 1.0 + (score / 100) * upvote_ratio)
            
        elif item['source'] == 'news':
            # Новости имеют высокий базовый вес
            weight = 2.0
        
        return weight
    
    def _determine_trend(self, sentiment: float) -> str:
        """Определение тренда на основе настроений"""
        if sentiment > self.config.sentiment_threshold:
            return 'bullish'
        elif sentiment < -self.config.sentiment_threshold:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_strength(self, sentiment: float, confidence: float) -> str:
        """Определение силы сигнала"""
        strength_score = abs(sentiment) * confidence
        
        if strength_score > 0.6:
            return 'strong'
        elif strength_score > 0.3:
            return 'moderate'
        else:
            return 'weak'
    
    def get_sentiment_history(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Получение истории настроений"""
        # Фильтрация кэша по времени и символу
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history_data = []
        for cache_key, sentiment_data in self.sentiment_cache.items():
            if (sentiment_data.symbol == symbol and 
                sentiment_data.timestamp >= cutoff_time):
                
                history_data.append({
                    'timestamp': sentiment_data.timestamp,
                    'sentiment': sentiment_data.overall_sentiment,
                    'confidence': sentiment_data.confidence,
                    'volume': sentiment_data.volume,
                    'trend': sentiment_data.trend,
                    'strength': sentiment_data.strength
                })
        
        return pd.DataFrame(history_data).sort_values('timestamp')
    
    def calculate_sentiment_momentum(self, symbol: str, periods: int = 5) -> float:
        """Вычисление моментума настроений"""
        history = self.get_sentiment_history(symbol, hours=periods)
        
        if len(history) < 2:
            return 0.0
        
        # Простой моментум как разность между последним и первым значением
        recent_sentiment = history['sentiment'].iloc[-1]
        old_sentiment = history['sentiment'].iloc[0]
        
        return recent_sentiment - old_sentiment

# Пример использования
async def demo_sentiment_analysis():
    """Демонстрация анализа настроений"""
    
    print("📊 ДЕМОНСТРАЦИЯ АНАЛИЗА НАСТРОЕНИЙ")
    print("=" * 50)
    
    # Конфигурация
    config = SentimentConfig(
        sources=['news'],  # Только новости для демонстрации
        max_posts_per_source=20
    )
    
    # Создание агрегатора
    aggregator = SentimentAggregator(config)
    
    # Анализ настроений для популярных символов
    symbols = ['AAPL', 'TSLA', 'BTC', 'EURUSD']
    
    print("🔍 Анализ настроений для символов:", symbols)
    
    try:
        results = await aggregator.analyze_sentiment_for_symbols(symbols)
        
        for symbol, sentiment_data in results.items():
            print(f"\n📈 {symbol}:")
            print(f"   Общее настроение: {sentiment_data.overall_sentiment:.3f}")
            print(f"   Тренд: {sentiment_data.trend}")
            print(f"   Сила: {sentiment_data.strength}")
            print(f"   Уверенность: {sentiment_data.confidence:.3f}")
            print(f"   Объем упоминаний: {sentiment_data.volume}")
            
            if sentiment_data.sources:
                print(f"   По источникам:")
                for source, score in sentiment_data.sources.items():
                    print(f"     {source}: {score:.3f}")
        
        # Демонстрация анализа отдельного текста
        print(f"\n🔬 Анализ примерного текста:")
        sample_text = "Apple stock is showing strong bullish momentum after great earnings. $AAPL to the moon! 🚀"
        
        if NLTK_AVAILABLE or TRANSFORMERS_AVAILABLE:
            sentiment, confidence = await aggregator.sentiment_analyzer.analyze_text(sample_text)
            symbols_found = aggregator.sentiment_analyzer.extract_symbols(sample_text)
            
            print(f"   Текст: {sample_text}")
            print(f"   Настроение: {sentiment:.3f}")
            print(f"   Уверенность: {confidence:.3f}")
            print(f"   Найденные символы: {symbols_found}")
        
    except Exception as e:
        print(f"❌ Ошибка при анализе настроений: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Демонстрация анализа настроений завершена")

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Запуск демонстрации
    asyncio.run(demo_sentiment_analysis())

