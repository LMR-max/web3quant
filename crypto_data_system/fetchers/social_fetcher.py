"""
社交媒体数据获取器模块
用于获取加密货币相关的社交媒体数据（Twitter, Reddit, Telegram等）
"""

import time
import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import logging

try:
    from .base_fetcher import BaseFetcher, AsyncFetcher
    from ..data_models import SocialSentimentData, SocialPostData, SocialSentiment
except ImportError:
    # 如果直接运行，使用简单导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fetchers.base_fetcher import BaseFetcher, AsyncFetcher
    from data_models import SocialSentimentData, SocialPostData, SocialSentiment


# ==================== 社交媒体配置 ====================

class SocialConfig:
    """社交媒体配置"""
    
    # Twitter配置
    TWITTER_API_KEY = None
    TWITTER_API_SECRET = None
    TWITTER_ACCESS_TOKEN = None
    TWITTER_ACCESS_SECRET = None
    TWITTER_BEARER_TOKEN = None
    
    # Reddit配置
    REDDIT_CLIENT_ID = None
    REDDIT_CLIENT_SECRET = None
    REDDIT_USER_AGENT = None
    REDDIT_USERNAME = None
    REDDIT_PASSWORD = None
    
    # Telegram配置
    TELEGRAM_API_ID = None
    TELEGRAM_API_HASH = None
    TELEGRAM_BOT_TOKEN = None
    
    # 通用配置
    RATE_LIMIT = 1.0  # 请求间隔（秒）
    MAX_RETRIES = 3
    TIMEOUT = 30
    PROXY_URL = None
    ENABLE_CACHE = True
    CACHE_TTL = 3600  # 缓存时间（秒）


# ==================== 社交媒体数据模型 ====================

class SocialData:
    """社交媒体数据容器"""
    
    def __init__(self):
        self.posts = []
        self.metrics = {}  # Dict[str, SocialSentimentData]
        self.sentiment = None
        self.trends = []
        self.influencers = []
        self.last_update = datetime.now()
    
    def add_post(self, post: SocialPostData):
        """添加帖子"""
        self.posts.append(post)
    
    def update_metrics(self, symbol: str, metrics: Any):
        """更新指标"""
        self.metrics[symbol] = metrics
    
    def set_sentiment(self, sentiment: SocialSentimentData):
        """设置情绪"""
        self.sentiment = sentiment
    
    def add_trend(self, trend: Dict[str, Any]):
        """添加趋势"""
        self.trends.append(trend)
    
    def add_influencer(self, influencer: Dict[str, Any]):
        """添加影响者"""
        self.influencers.append(influencer)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'post_count': len(self.posts),
            'metrics': {k: (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in self.metrics.items()},
            'sentiment': self.sentiment.to_dict() if self.sentiment else None,
            'trends': self.trends,
            'influencers': self.influencers,
            'last_update': self.last_update.isoformat()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.posts:
            return pd.DataFrame()
        
        data = []
        for post in self.posts:
            # support both old and new object attrs just in case, but prefer new
            likes = getattr(post, 'like_count', getattr(post, 'likes', 0))
            shares = getattr(post, 'share_count', getattr(post, 'shares', 0))
            comments = getattr(post, 'reply_count', getattr(post, 'comments', 0))
            
            data.append({
                'id': post.post_id,
                'platform': getattr(post, 'platform', 'unknown'),
                'symbol': getattr(post, 'symbol', ''),
                'user': post.author_id,
                'username': post.author_name,
                'text': post.text[:100],
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'timestamp': post.timestamp,
                'sentiment': getattr(post, 'sentiment', 'neutral'), # derived usually
                'sentiment_score': post.sentiment_score,
                'url': post.urls[0] if post.urls else ''
            })
        
        return pd.DataFrame(data)


# ==================== 基础社交媒体获取器 ====================

class BaseSocialFetcher(BaseFetcher):
    """
    基础社交媒体获取器
    支持Twitter、Reddit、Telegram等平台
    """
    
    def __init__(self, 
                 name: str = "social_fetcher",
                 platform: str = "twitter",
                 config: Optional[Dict] = None,
                 **kwargs):
        """
        初始化社交媒体获取器
        
        参数:
            name: 获取器名称
            platform: 平台名称 (twitter, reddit, telegram)
            config: 配置字典
            **kwargs: 额外参数
        """
        super().__init__(exchange="social", market_type=platform, **kwargs)
        
        self.platform = platform
        self.social_config = self._load_social_config(config)
        
        # 平台客户端
        self.client = None
        self.is_authenticated = False
        
        # 社交媒体特定统计
        self.post_count = 0
        self.user_count = 0
        self.sentiment_analyzed = 0
        
        self.logger.info(f"初始化社交媒体获取器: {name}, 平台: {platform}")

    def get_available_symbols(self) -> List[str]:
        """前端用于“加载交易对”的可选项。

        这里的 symbol 表示加密货币符号（如 BTC/ETH），用于社交热度/情绪聚合。
        """
        return [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'TRX', 'AVAX', 'DOT',
            'MATIC', 'LINK', 'LTC', 'BCH', 'ATOM', 'UNI', 'AAVE', 'SUI', 'OP', 'ARB'
        ]
    
    def _load_social_config(self, config: Optional[Dict]) -> Dict:
        """加载社交媒体配置"""
        if config is None:
            config = {}
        
        # 加载默认配置
        default_config = {
            'rate_limit': 1.0,
            'max_retries': 3,
            'timeout': 30,
            'proxy_url': None,
            'enable_cache': True,
            'cache_ttl': 3600,
            'max_posts': 100,
            'max_users': 50,
            'min_likes': 1,
            'min_followers': 100,
            'language': 'en',
            'keywords': ['crypto', 'bitcoin', 'ethereum', 'blockchain'],
            'symbols': ['BTC', 'ETH', 'XRP', 'ADA', 'DOGE'],
            'sentiment_enabled': True,
            'trend_analysis': True,
            'influencer_detection': True,
        }
        
        # 合并配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _init_exchange(self):
        """初始化社交媒体平台连接（实现抽象方法）"""
        try:
            self.logger.info(f"初始化社交媒体平台: {self.platform}")
            # 社交媒体平台初始化在 authenticate 中完成
            self.authenticate()
        except Exception as e:
            self.logger.error(f"初始化社交媒体平台失败: {e}")
    
    def authenticate(self) -> bool:
        """
        认证社交媒体平台
        
        返回:
            认证是否成功
        """
        try:
            if self.platform == "twitter":
                success = self._authenticate_twitter()
            elif self.platform == "reddit":
                success = self._authenticate_reddit()
            elif self.platform == "telegram":
                success = self._authenticate_telegram()
            else:
                self.logger.error(f"不支持的平台: {self.platform}")
                return False
            
            if success:
                self.is_authenticated = True
                self.logger.info(f"{self.platform} 认证成功")
            else:
                self.logger.error(f"{self.platform} 认证失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"认证失败: {e}")
            return False
    
    def _authenticate_twitter(self) -> bool:
        """认证Twitter"""
        try:
            # 尝试导入tweepy
            import tweepy
            
            # 检查配置
            if not self.social_config.get('twitter_api_key'):
                self.logger.warning("Twitter API密钥未配置，使用有限功能")
                return True  # 部分功能可能不需要认证
            
            # 创建认证对象
            auth = tweepy.OAuth1UserHandler(
                self.social_config.get('twitter_api_key'),
                self.social_config.get('twitter_api_secret'),
                self.social_config.get('twitter_access_token'),
                self.social_config.get('twitter_access_secret')
            )
            
            # 创建API客户端
            self.client = tweepy.API(auth, wait_on_rate_limit=True)
            
            # 测试连接
            if self.social_config.get('twitter_bearer_token'):
                # 使用v2 API
                self.client_v2 = tweepy.Client(
                    bearer_token=self.social_config.get('twitter_bearer_token'),
                    wait_on_rate_limit=True
                )
            
            return True
            
        except ImportError:
            self.logger.error("tweepy库未安装，请使用: pip install tweepy")
            return False
        except Exception as e:
            self.logger.error(f"Twitter认证失败: {e}")
            return False
    
    def _authenticate_reddit(self) -> bool:
        """认证Reddit"""
        try:
            # 尝试导入praw
            import praw
            
            # 检查配置
            if not self.social_config.get('reddit_client_id'):
                self.logger.warning("Reddit客户端ID未配置，使用有限功能")
                return True
            
            # 创建Reddit实例
            self.client = praw.Reddit(
                client_id=self.social_config.get('reddit_client_id'),
                client_secret=self.social_config.get('reddit_client_secret'),
                user_agent=self.social_config.get('reddit_user_agent', 'CryptoSocialFetcher/1.0'),
                username=self.social_config.get('reddit_username'),
                password=self.social_config.get('reddit_password')
            )
            
            return True
            
        except ImportError:
            self.logger.error("praw库未安装，请使用: pip install praw")
            return False
        except Exception as e:
            self.logger.error(f"Reddit认证失败: {e}")
            return False
    
    def _authenticate_telegram(self) -> bool:
        """认证Telegram"""
        try:
            # 尝试导入telethon
            from telethon import TelegramClient
            
            # 检查配置
            if not self.social_config.get('telegram_api_id'):
                self.logger.warning("Telegram API ID未配置，使用有限功能")
                return True
            
            # 创建Telegram客户端
            self.client = TelegramClient(
                'crypto_social_fetcher',
                self.social_config.get('telegram_api_id'),
                self.social_config.get('telegram_api_hash')
            )
            
            # 启动客户端（需要在外部调用start()）
            return True
            
        except ImportError:
            self.logger.error("telethon库未安装，请使用: pip install telethon")
            return False
        except Exception as e:
            self.logger.error(f"Telegram认证失败: {e}")
            return False
    
    def fetch_posts(self, 
                   query: str = None,
                   symbol: str = None,
                   since: Optional[datetime] = None,
                   until: Optional[datetime] = None,
                   limit: int = 100,
                   **kwargs) -> List[SocialPostData]:
        """
        获取社交媒体帖子
        
        参数:
            query: 搜索查询
            symbol: 加密货币符号
            since: 开始时间
            until: 结束时间
            limit: 最大帖子数量
            **kwargs: 额外参数
            
        返回:
            帖子列表
        """
        # 构建查询
        if not query and symbol:
            query = self._build_query_from_symbol(symbol)
        
        if not query:
            query = " ".join(self.social_config.get('keywords', []))
        
        # 调用平台特定方法
        if self.platform == "twitter":
            return self._fetch_twitter_posts(query, since, until, limit, **kwargs)
        elif self.platform == "reddit":
            return self._fetch_reddit_posts(query, since, until, limit, **kwargs)
        elif self.platform == "telegram":
            return self._fetch_telegram_posts(query, since, until, limit, **kwargs)
        else:
            self.logger.error(f"不支持的平台: {self.platform}")
            return []
    
    def _build_query_from_symbol(self, symbol: str) -> str:
        """从符号构建查询"""
        # 加密货币符号映射
        crypto_terms = {
            'BTC': ['bitcoin', 'btc', '#bitcoin', '#btc'],
            'ETH': ['ethereum', 'eth', '#ethereum', '#eth'],
            'XRP': ['ripple', 'xrp', '#ripple', '#xrp'],
            'ADA': ['cardano', 'ada', '#cardano', '#ada'],
            'DOGE': ['dogecoin', 'doge', '#dogecoin', '#doge'],
            'SOL': ['solana', 'sol', '#solana', '#sol'],
            'DOT': ['polkadot', 'dot', '#polkadot', '#dot'],
            'LINK': ['chainlink', 'link', '#chainlink', '#link'],
            'BNB': ['binance', 'bnb', '#binance', '#bnb'],
            'USDT': ['tether', 'usdt', '#tether', '#usdt'],
        }
        
        terms = crypto_terms.get(symbol.upper(), [symbol.lower()])
        return f"({' OR '.join(terms)}) crypto"
    
    def _fetch_twitter_posts(self, query: str, since: datetime, until: datetime, 
                           limit: int, **kwargs) -> List[SocialPostData]:
        """获取Twitter帖子"""
        posts = []
        
        try:
            if not self.client:
                self.logger.warning("Twitter客户端未初始化")
                return posts
            
            # 计算最大ID（用于分页）
            max_id = None
            tweet_count = 0
            
            while tweet_count < limit:
                try:
                    # 搜索推文
                    tweets = self.client.search_tweets(
                        q=query,
                        count=min(100, limit - tweet_count),
                        since_id=max_id,
                        tweet_mode='extended'
                    )
                    
                    if not tweets:
                        break
                    
                    for tweet in tweets:
                        # 转换为SocialPostData
                        post = self._tweet_to_data(tweet)
                        posts.append(post)
                        tweet_count += 1
                        
                        # 更新max_id用于分页
                        if max_id is None or tweet.id < max_id:
                            max_id = tweet.id - 1
                    
                    # 频率限制
                    time.sleep(self.social_config.get('rate_limit', 1.0))
                    
                except Exception as e:
                    self.logger.error(f"获取推文失败: {e}")
                    break
            
            self.post_count += len(posts)
            self.logger.info(f"获取 {len(posts)} 条Twitter帖子")
            
        except Exception as e:
            self.logger.error(f"Twitter获取失败: {e}")
        
        return posts
    
    def _tweet_to_data(self, tweet) -> SocialPostData:
        """转换Tweet为SocialPostData"""
        # 提取文本
        text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
        
        # 分析情绪
        sentiment, sentiment_score = self._analyze_sentiment(text)
        
        # 提取加密货币符号
        symbols = self._extract_symbols(text)
        
        # 创建帖子对象
        post = SocialPostData(
            post_id=str(tweet.id),
            timestamp=datetime.fromtimestamp(tweet.created_at_in_seconds) 
                       if hasattr(tweet, 'created_at_in_seconds') 
                       else pd.Timestamp(tweet.created_at),
            symbol=symbols[0] if symbols else '', # primary symbol
            exchange='twitter', # used as platform
            market_type='social',
            text=text,
            author_id=str(tweet.user.id),
            author_name=tweet.user.screen_name,
            like_count=tweet.favorite_count,
            share_count=tweet.retweet_count, # retweet as share
            reply_count=0,  # Twitter API simple object doesn't always have reply count easily
            sentiment_score=sentiment_score,
            urls=[f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"],
            hashtags=[hashtag['text'] for hashtag in tweet.entities.get('hashtags', [])],
            mentions=[mention['screen_name'] for mention in tweet.entities.get('user_mentions', [])],
            extra_info={'raw_data': tweet._json if hasattr(tweet, '_json') else str(tweet)}
        )
        
        return post
    
    def _fetch_reddit_posts(self, query: str, since: datetime, until: datetime,
                          limit: int, **kwargs) -> List[SocialPostData]:
        """获取Reddit帖子"""
        posts = []
        
        try:
            if not self.client:
                self.logger.warning("Reddit客户端未初始化")
                return posts
            
            # 搜索subreddit或使用通用搜索
            subreddit_name = kwargs.get('subreddit', 'all')
            sort_by = kwargs.get('sort_by', 'relevance')
            time_filter = kwargs.get('time_filter', 'all')
            
            # 搜索帖子
            search_results = self.client.subreddit(subreddit_name).search(
                query=query,
                sort=sort_by,
                time_filter=time_filter,
                limit=limit
            )
            
            for submission in search_results:
                # 转换为SocialPostData
                post = self._reddit_submission_to_data(submission)
                posts.append(post)
            
            self.post_count += len(posts)
            self.logger.info(f"获取 {len(posts)} 条Reddit帖子")
            
        except Exception as e:
            self.logger.error(f"Reddit获取失败: {e}")
        
        return posts
    
    def _reddit_submission_to_data(self, submission) -> SocialPostData:
        """转换Reddit提交为SocialPostData"""
        # 分析情绪
        text_content = submission.title + " " + submission.selftext
        sentiment, sentiment_score = self._analyze_sentiment(text_content)
        
        # 提取加密货币符号
        symbols = self._extract_symbols(text_content)
        
        # 创建帖子对象
        post = SocialPostData(
            post_id=str(submission.id),
            timestamp=pd.Timestamp(submission.created_utc, unit='s'),
            symbol=symbols[0] if symbols else '',
            exchange='reddit',
            market_type='social',
            text=submission.title + "\n\n" + submission.selftext,
            author_id=str(submission.author),
            author_name=str(submission.author),
            like_count=submission.score,
            share_count=0,
            reply_count=submission.num_comments,
            sentiment_score=sentiment_score,
            urls=[f"https://reddit.com{submission.permalink}"],
            extra_info={
                'subreddit': submission.subreddit.display_name,
                'upvote_ratio': submission.upvote_ratio,
                'sentiment_label': sentiment
            }
        )
        
        return post
    
    def _fetch_telegram_posts(self, query: str, since: datetime, until: datetime,
                            limit: int, **kwargs) -> List[SocialPostData]:
        """获取Telegram消息"""
        posts = []
        
        try:
            if not self.client:
                self.logger.warning("Telegram客户端未初始化")
                return posts
            
            # 获取频道或群组
            channel = kwargs.get('channel')
            group = kwargs.get('group')
            
            if not channel and not group:
                self.logger.warning("未指定频道或群组")
                return posts
            
            # 这里需要实际的Telegram客户端实现
            # 由于Telegram API的限制，这里只提供框架
            
            self.logger.warning("Telegram获取功能需要完整实现telethon客户端")
            
        except Exception as e:
            self.logger.error(f"Telegram获取失败: {e}")
        
        return posts
    
    def _analyze_sentiment(self, text: str) -> tuple:
        """
        分析文本情绪
        
        参数:
            text: 文本
            
        返回:
            (情绪, 分数) 元组
        """
        if not self.social_config.get('sentiment_enabled', True):
            return "neutral", 0.0
        
        try:
            # 简单的基于关键词的情绪分析
            positive_words = [
                'bullish', 'moon', '🚀', 'rocket', 'buy', 'long', '上涨', '涨',
                'good', 'great', 'excellent', 'amazing', 'awesome', 'profit',
                'win', 'gain', 'success', '突破', '新高', '暴涨'
            ]
            
            negative_words = [
                'bearish', 'dump', 'crash', 'sell', 'short', '下跌', '跌',
                'bad', 'terrible', 'awful', 'horrible', 'loss', 'lose',
                'fail', 'failure', '破产', '归零', '暴跌', '崩盘'
            ]
            
            # 计算情绪分数
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word.lower() in text_lower)
            negative_count = sum(1 for word in negative_words if word.lower() in text_lower)
            
            total = positive_count + negative_count
            if total == 0:
                return "neutral", 0.0
            
            score = (positive_count - negative_count) / total
            
            if score > 0.2:
                sentiment = "positive"
            elif score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            self.sentiment_analyzed += 1
            return sentiment, score
            
        except Exception as e:
            self.logger.error(f"情绪分析失败: {e}")
            return "neutral", 0.0
    
    def _extract_symbols(self, text: str) -> List[str]:
        """
        从文本中提取加密货币符号
        
        参数:
            text: 文本
            
        返回:
            符号列表
        """
        symbols = []
        
        # 常见加密货币符号模式
        patterns = [
            r'\$([A-Z]{2,5})\b',  # $BTC, $ETH
            r'\b([A-Z]{2,5})\b',  # BTC, ETH
            r'#([A-Z]{2,5})\b',   # #BTC, #ETH
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.extend(matches)
        
        # 去重
        symbols = list(set(symbols))
        
        # 过滤常见非加密货币符号
        common_words = ['THE', 'AND', 'FOR', 'ARE', 'YOU', 'ALL', 'NOT', 'BUT', 'HAS', 'WAS']
        symbols = [s for s in symbols if s not in common_words and len(s) <= 5]
        
        return symbols
    
    def fetch_metrics(self, 
                     symbol: str,
                     period: str = "24h",
                     **kwargs) -> SocialSentimentData:
        """
        获取社交媒体指标 (返回 SocialSentimentData)
        
        参数:
            symbol: 加密货币符号
            period: 时间段 (24h, 7d, 30d)
            **kwargs: 额外参数
            
        返回:
            社交媒体情绪数据
        """
        try:
            # 获取相关帖子
            posts = self.fetch_posts(symbol=symbol, limit=100, **kwargs)
            
            if not posts:
                return SocialSentimentData(
                    timestamp=pd.Timestamp.now(),
                    symbol=symbol,
                    exchange="social",
                    market_type=self.platform,
                    platform=self.platform,
                    keyword=symbol
                )
            
            # 计算指标
            total_likes = sum(p.like_count for p in posts)
            total_shares = sum(p.share_count for p in posts)
            total_comments = sum(p.reply_count for p in posts)
            
            # 情绪分数
            sentiment_scores = [p.sentiment_score for p in posts if p.sentiment_score is not None]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            # 统计正负面
            positive_count = sum(1 for p in posts if p.sentiment_score > 0.2)
            negative_count = sum(1 for p in posts if p.sentiment_score < -0.2)
            neutral_count = len(posts) - positive_count - negative_count

            # 参与率
            engagement_rate = (total_likes + total_shares + total_comments) / len(posts) if posts else 0.0
            
            # Top influencers extraction (simplified)
            # sort by engagement
            sorted_posts = sorted(posts, key=lambda x: (x.like_count + x.share_count), reverse=True)
            top_influencers = []
            seen_authors = set()
            for p in sorted_posts:
                if p.author_name and p.author_name not in seen_authors:
                    top_influencers.append({
                        'author_name': p.author_name,
                        'author_id': p.author_id,
                        'engagement': p.like_count + p.share_count
                    })
                    seen_authors.add(p.author_name)
                    if len(top_influencers) >= 5:
                        break

            # 创建指标对象
            metric = SocialSentimentData(
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                exchange="social",
                market_type=self.platform,
                platform=self.platform,
                keyword=symbol,
                sentiment_score=avg_sentiment,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                total_mentions=len(posts),
                engagement_rate=engagement_rate,
                top_influencers=top_influencers
            )
            
            self.logger.info(f"获取 {symbol} 社交媒体指标: {len(posts)} 条帖子, 情绪分: {avg_sentiment:.2f}")
            return metric
            
        except Exception as e:
            self.logger.error(f"获取指标失败: {e}")
            return SocialSentimentData(
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                exchange="social",
                market_type=self.platform,
                platform=self.platform,
                keyword=symbol
            )
    
    def fetch_trends(self, 
                    limit: int = 10,
                    **kwargs) -> List[Dict[str, Any]]:
        """
        获取社交媒体趋势
        
        参数:
            limit: 最大趋势数量
            **kwargs: 额外参数
            
        返回:
            趋势列表
        """
        trends = []
        
        try:
            if self.platform == "twitter":
                # 获取Twitter趋势
                if hasattr(self, 'client_v2') and self.client_v2:
                    # 使用Twitter API v2获取趋势
                    pass
                    
            elif self.platform == "reddit":
                # 获取Reddit热门话题
                if self.client:
                    for subreddit_name in ['cryptocurrency', 'bitcoin', 'ethereum']:
                        try:
                            subreddit = self.client.subreddit(subreddit_name)
                            for post in subreddit.hot(limit=5):
                                trends.append({
                                    'platform': 'reddit',
                                    'title': post.title,
                                    'subreddit': subreddit_name,
                                    'score': post.score,
                                    'comments': post.num_comments,
                                    'url': f"https://reddit.com{post.permalink}",
                                    'symbols': self._extract_symbols(post.title + " " + post.selftext)
                                })
                        except Exception as e:
                            self.logger.error(f"获取subreddit {subreddit_name} 失败: {e}")
            
            self.logger.info(f"获取 {len(trends)} 条趋势")
            
        except Exception as e:
            self.logger.error(f"获取趋势失败: {e}")
        
        return trends[:limit]
    
    def fetch_influencers(self, 
                         symbol: str = None,
                         limit: int = 10,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        获取影响者
        
        参数:
            symbol: 加密货币符号
            limit: 最大影响者数量
            **kwargs: 额外参数
            
        返回:
            影响者列表
        """
        influencers = []
        
        try:
            if self.platform == "twitter":
                # 这里可以实现Twitter影响者分析
                pass
                
            elif self.platform == "reddit":
                # 分析Reddit用户
                if self.client and symbol:
                    # 搜索相关帖子
                    posts = self.fetch_posts(symbol=symbol, limit=50, **kwargs)
                    
                    # 统计用户活跃度
                    user_stats = {}
                    for post in posts:
                        user_id = post.user_id
                        if user_id not in user_stats:
                            user_stats[user_id] = {
                                'username': post.username,
                                'post_count': 0,
                                'total_likes': 0,
                                'total_comments': 0,
                                'avg_sentiment': 0.0
                            }
                        
                        user_stats[user_id]['post_count'] += 1
                        user_stats[user_id]['total_likes'] += post.likes
                        user_stats[user_id]['total_comments'] += post.comments
                    
                    # 转换为影响者列表
                    for user_id, stats in user_stats.items():
                        if stats['post_count'] >= 2:  # 至少2个帖子
                            influencers.append({
                                'platform': 'reddit',
                                'user_id': user_id,
                                'username': stats['username'],
                                'post_count': stats['post_count'],
                                'total_engagement': stats['total_likes'] + stats['total_comments'],
                                'symbol': symbol
                            })
                    
                    # 按参与度排序
                    influencers.sort(key=lambda x: x['total_engagement'], reverse=True)
            
            self.user_count += len(influencers)
            self.logger.info(f"获取 {len(influencers)} 个影响者")
            
        except Exception as e:
            self.logger.error(f"获取影响者失败: {e}")
        
        return influencers[:limit]
    
    def analyze_sentiment_over_time(self,
                                   symbol: str,
                                   days: int = 7,
                                   **kwargs) -> Dict[str, Any]:
        """
        分析一段时间内的情绪变化
        
        参数:
            symbol: 加密货币符号
            days: 天数
            **kwargs: 额外参数
            
        返回:
            情绪分析结果
        """
        results = {
            'symbol': symbol,
            'platform': self.platform,
            'days': days,
            'daily_sentiment': [],
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'total_posts': 0,
            'start_date': datetime.now() - timedelta(days=days),
            'end_date': datetime.now()
        }
        
        try:
            # 按天获取数据
            for i in range(days):
                day_start = datetime.now() - timedelta(days=i+1)
                day_end = datetime.now() - timedelta(days=i)
                
                # 获取该天的帖子
                posts = self.fetch_posts(
                    symbol=symbol,
                    since=day_start,
                    until=day_end,
                    limit=50,
                    **kwargs
                )
                
                if posts:
                    # 计算当天情绪
                    sentiment_scores = [p.sentiment_score for p in posts if p.sentiment_score is not None]
                    day_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
                    
                    results['daily_sentiment'].append({
                        'date': day_start.strftime('%Y-%m-%d'),
                        'post_count': len(posts),
                        'sentiment_score': day_sentiment,
                        'sentiment': 'positive' if day_sentiment > 0.2 else 'negative' if day_sentiment < -0.2 else 'neutral'
                    })
            
            # 计算总体情绪
            if results['daily_sentiment']:
                total_score = sum(day['sentiment_score'] for day in results['daily_sentiment'])
                avg_score = total_score / len(results['daily_sentiment'])
                
                results['sentiment_score'] = avg_score
                results['total_posts'] = sum(day['post_count'] for day in results['daily_sentiment'])
                results['overall_sentiment'] = 'positive' if avg_score > 0.2 else 'negative' if avg_score < -0.2 else 'neutral'
            
            self.logger.info(f"分析 {symbol} {days}天情绪变化: {results['total_posts']} 条帖子")
            
        except Exception as e:
            self.logger.error(f"情绪时间分析失败: {e}")
        
        return results
    
    def fetch_comprehensive_data(self,
                               symbol: str,
                               **kwargs) -> SocialData:
        """
        获取全面的社交媒体数据
        
        参数:
            symbol: 加密货币符号
            **kwargs: 额外参数
            
        返回:
            社交媒体数据容器
        """
        social_data = SocialData()
        
        try:
            # 获取帖子
            posts = self.fetch_posts(symbol=symbol, **kwargs)
            for post in posts:
                social_data.add_post(post)
            
            # 获取指标
            metrics = self.fetch_metrics(symbol=symbol, **kwargs)
            social_data.update_metrics(symbol, metrics)
            
            # 获取趋势
            trends = self.fetch_trends(**kwargs)
            for trend in trends:
                social_data.add_trend(trend)
            
            # 获取影响者
            influencers = self.fetch_influencers(symbol=symbol, **kwargs)
            for influencer in influencers:
                social_data.add_influencer(influencer)
            
            # 分析情绪
            sentiment_analysis = self.analyze_sentiment_over_time(symbol=symbol, **kwargs)
            # 计算正负中性计数（使用 sentiment_score）
            positive_count = sum(1 for p in posts if getattr(p, 'sentiment_score', None) is not None and p.sentiment_score > 0.2)
            negative_count = sum(1 for p in posts if getattr(p, 'sentiment_score', None) is not None and p.sentiment_score < -0.2)
            neutral_count = len(posts) - positive_count - negative_count

            sentiment = SocialSentiment(
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                platform=self.platform,
                overall_sentiment=sentiment_analysis.get('overall_sentiment', 'neutral'),
                sentiment_score=sentiment_analysis.get('sentiment_score', 0.0),
                confidence=0.8,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                total_mentions=len(posts),
                analysis_period=f"{sentiment_analysis.get('days', 0)}天"
            )
            social_data.set_sentiment(sentiment)
            
            self.logger.info(f"获取 {symbol} 全面社交媒体数据: {len(posts)} 条帖子")
            
        except Exception as e:
            self.logger.error(f"获取全面数据失败: {e}")
        
        return social_data
    
    # 实现抽象方法（对于社交媒体不适用，返回空数据）
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", since=None, limit=None, **kwargs):
        """获取K线数据（对于社交媒体不适用）"""
        self.logger.warning(f"社交媒体获取器不支持OHLCV数据")
        return []
    
    def fetch_orderbook(self, symbol: str, limit=None, **kwargs):
        """获取订单簿数据（对于社交媒体不适用）"""
        self.logger.warning(f"社交媒体获取器不支持订单簿数据")
        return None
    
    def fetch_trades(self, symbol: str, since=None, limit=None, **kwargs):
        """获取成交数据（对于社交媒体不适用）"""
        self.logger.warning(f"社交媒体获取器不支持成交数据")
        return []
    
    def get_available_symbols(self) -> List[str]:
        """获取可用符号（社交媒体关注的所有加密货币）"""
        return self.social_config.get('symbols', [])
    
    def get_status(self) -> Dict[str, Any]:
        """获取获取器状态"""
        status = super().get_status()
        status.update({
            'platform': self.platform,
            'is_authenticated': self.is_authenticated,
            'post_count': self.post_count,
            'user_count': self.user_count,
            'sentiment_analyzed': self.sentiment_analyzed,
            'social_config': {
                'max_posts': self.social_config.get('max_posts'),
                'keywords': self.social_config.get('keywords'),
                'symbols': self.social_config.get('symbols'),
                'sentiment_enabled': self.social_config.get('sentiment_enabled'),
            }
        })
        return status
    
    def initialize(self):
        """初始化获取器"""
        if not self.is_authenticated:
            success = self.authenticate()
            if success:
                self.is_initialized = True
                return True
            return False
        return True
    
    def close(self):
        """关闭获取器"""
        if self.client:
            # 清理客户端资源
            if self.platform == "telegram" and hasattr(self.client, 'disconnect'):
                try:
                    self.client.disconnect()
                except:
                    pass
        
        super().close()


# ==================== 多平台社交媒体获取器 ====================

class MultiPlatformSocialFetcher:
    """
    多平台社交媒体获取器
    同时从多个平台获取数据
    """
    
    def __init__(self, platforms: List[str] = None, config: Optional[Dict] = None):
        """
        初始化多平台获取器
        
        参数:
            platforms: 平台列表
            config: 配置字典
        """
        self.platforms = platforms or ["twitter", "reddit"]
        self.config = config or {}
        
        # 初始化日志
        self.logger = logging.getLogger("multi_platform_social_fetcher")
        
        # 平台获取器字典
        self.fetchers = {}
        
        # 初始化状态
        self.is_initialized = False
        
        self.logger.info(f"初始化多平台社交媒体获取器: {', '.join(self.platforms)}")
    
    def initialize(self):
        """初始化所有平台获取器"""
        for platform in self.platforms:
            try:
                fetcher = BaseSocialFetcher(
                    name=f"social_{platform}",
                    platform=platform,
                    config=self.config.get(platform, {})
                )
                
                if fetcher.initialize():
                    self.fetchers[platform] = fetcher
                    self.logger.info(f"初始化 {platform} 平台成功")
                else:
                    self.logger.warning(f"初始化 {platform} 平台失败")
                    
            except Exception as e:
                self.logger.error(f"创建 {platform} 获取器失败: {e}")
        
        self.is_initialized = len(self.fetchers) > 0
        return self.is_initialized
    
    def fetch_multi_platform_data(self,
                                symbol: str,
                                **kwargs) -> Dict[str, SocialData]:
        """
        从多个平台获取数据
        
        参数:
            symbol: 加密货币符号
            **kwargs: 额外参数
            
        返回:
            按平台组织的数据字典
        """
        results = {}
        
        for platform, fetcher in self.fetchers.items():
            try:
                data = fetcher.fetch_comprehensive_data(symbol, **kwargs)
                results[platform] = data
                self.logger.info(f"从 {platform} 获取数据成功: {len(data.posts)} 条帖子")
            except Exception as e:
                self.logger.error(f"从 {platform} 获取数据失败: {e}")
                results[platform] = None
        
        return results
    
    def aggregate_sentiment(self,
                          symbol: str,
                          **kwargs) -> Dict[str, Any]:
        """
        聚合多个平台的情绪分析
        
        参数:
            symbol: 加密货币符号
            **kwargs: 额外参数
            
        返回:
            聚合的情绪分析结果
        """
        platform_data = self.fetch_multi_platform_data(symbol, **kwargs)
        
        # 聚合情绪
        total_sentiment_score = 0.0
        total_posts = 0
        platform_sentiments = {}
        
        for platform, data in platform_data.items():
            if data and data.sentiment:
                platform_sentiments[platform] = {
                    'sentiment': data.sentiment.overall_sentiment,
                    'score': data.sentiment.sentiment_score,
                    'post_count': len(data.posts),
                    'confidence': data.sentiment.confidence
                }
                
                total_sentiment_score += data.sentiment.sentiment_score
                total_posts += len(data.posts)
        
        # 计算加权平均
        if platform_sentiments:
            avg_sentiment = total_sentiment_score / len(platform_sentiments)
            overall_sentiment = 'positive' if avg_sentiment > 0.2 else 'negative' if avg_sentiment < -0.2 else 'neutral'
        else:
            avg_sentiment = 0.0
            overall_sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'aggregated_score': avg_sentiment,
            'total_posts': total_posts,
            'platform_count': len(platform_sentiments),
            'platform_sentiments': platform_sentiments,
            'timestamp': datetime.now()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = {
            'platforms': self.platforms,
            'is_initialized': self.is_initialized,
            'fetcher_status': {}
        }
        
        for platform, fetcher in self.fetchers.items():
            status['fetcher_status'][platform] = fetcher.get_status()
        
        return status
    
    def close(self):
        """关闭所有获取器"""
        for platform, fetcher in self.fetchers.items():
            try:
                fetcher.close()
                self.logger.info(f"关闭 {platform} 获取器成功")
            except Exception as e:
                self.logger.error(f"关闭 {platform} 获取器失败: {e}")
        
        self.is_initialized = False


# ==================== 测试函数 ====================

def test_social_fetcher():
    """测试社交媒体获取器"""
    print("=" * 60)
    print("社交媒体获取器模块测试")
    print("=" * 60)
    
    # 测试基础功能
    print("\n1. 测试基础社交媒体获取器:")
    
    try:
        # 创建测试获取器（不使用真实API密钥）
        fetcher = BaseSocialFetcher(
            name="test_social",
            platform="twitter",
            config={
                'max_posts': 10,
                'keywords': ['bitcoin', 'crypto'],
                'symbols': ['BTC', 'ETH'],
                'sentiment_enabled': True
            }
        )
        
        print(f"✅ 获取器创建成功: {fetcher.name}")
        
        # 测试状态获取
        status = fetcher.get_status()
        print(f"✅ 状态获取成功: {status['platform']}")
        
        # 测试符号验证
        test_symbol = "BTC"
        is_valid = fetcher.validate_symbol(test_symbol)
        print(f"✅ 符号验证: {test_symbol} -> {is_valid}")
        
        # 测试符号格式化
        formatted = fetcher.format_symbol("btc-usdt")
        print(f"✅ 符号格式化: btc-usdt -> {formatted}")
        
        # 测试查询构建
        query = fetcher._build_query_from_symbol("BTC")
        print(f"✅ 查询构建: BTC -> {query}")
        
        # 测试情绪分析
        text = "Bitcoin is going to the moon! 🚀"
        sentiment, score = fetcher._analyze_sentiment(text)
        print(f"✅ 情绪分析: '{text}' -> {sentiment} ({score:.2f})")
        
        # 测试符号提取
        text_with_symbols = "I love $BTC and $ETH! #crypto"
        symbols = fetcher._extract_symbols(text_with_symbols)
        print(f"✅ 符号提取: '{text_with_symbols}' -> {symbols}")
        
        # 测试时间戳解析
        timestamp = datetime.now()
        parsed = fetcher.parse_timestamp(timestamp)
        print(f"✅ 时间戳解析: {timestamp} -> {parsed}")
        
        # 测试可用符号
        available_symbols = fetcher.get_available_symbols()
        print(f"✅ 可用符号: {available_symbols}")
        
        # 关闭获取器
        fetcher.close()
        print("✅ 获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试多平台获取器
    print("\n2. 测试多平台社交媒体获取器:")
    
    try:
        multi_fetcher = MultiPlatformSocialFetcher(
            platforms=["twitter", "reddit"],
            config={
                'twitter': {'max_posts': 5},
                'reddit': {'max_posts': 5}
            }
        )
        
        print(f"✅ 多平台获取器创建成功: {multi_fetcher.platforms}")
        
        # 初始化
        initialized = multi_fetcher.initialize()
        print(f"✅ 初始化结果: {initialized}")
        
        # 获取状态
        status = multi_fetcher.get_status()
        print(f"✅ 多平台状态: {len(status['fetcher_status'])} 个平台")
        
        # 注意：这里不实际获取数据，因为需要API密钥
        print("⚠️  实际数据获取需要API密钥配置")
        
        # 关闭
        multi_fetcher.close()
        print("✅ 多平台获取器关闭成功")
        
    except Exception as e:
        print(f"❌ 多平台测试失败: {e}")
    
    # 演示配置示例
    print("\n3. 配置示例:")
    print("Twitter配置示例:")
    print("""
    twitter_config = {
        'twitter_api_key': 'YOUR_API_KEY',
        'twitter_api_secret': 'YOUR_API_SECRET',
        'twitter_access_token': 'YOUR_ACCESS_TOKEN',
        'twitter_access_secret': 'YOUR_ACCESS_SECRET',
        'twitter_bearer_token': 'YOUR_BEARER_TOKEN',
        'max_posts': 100,
        'rate_limit': 1.0
    }
    """)
    
    print("Reddit配置示例:")
    print("""
    reddit_config = {
        'reddit_client_id': 'YOUR_CLIENT_ID',
        'reddit_client_secret': 'YOUR_CLIENT_SECRET',
        'reddit_user_agent': 'CryptoSocialFetcher/1.0',
        'max_posts': 100,
        'rate_limit': 1.0
    }
    """)
    
    print("\n✅ 社交媒体获取器模块测试完成")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_social_fetcher()