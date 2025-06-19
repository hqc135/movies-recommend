"""
人口统计学推荐模块
基于IMDB加权评分的简单推荐系统
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DemographicRecommender:
    """基于人口统计学的推荐器"""
    
    def __init__(self, config=None):
        self.config = config
        self.movies_df = None
        self.qualified_movies = None
        
    def fit(self, movies_df: pd.DataFrame):
        """训练推荐器"""
        self.movies_df = movies_df.copy()
        logger.info("训练人口统计学推荐器...")
        
        # 确保数值列是正确的数据类型
        numeric_columns = ['vote_count', 'vote_average']
        for col in numeric_columns:
            if col in self.movies_df.columns:
                self.movies_df[col] = pd.to_numeric(self.movies_df[col], errors='coerce').fillna(0)
        
        # 计算评分统计
        self.C = self.movies_df["vote_average"].mean()
        self.m = self.movies_df["vote_count"].quantile(0.9)
        
        # 筛选符合条件的电影 - 确保数值比较
        self.qualified_movies = self.movies_df.loc[
            pd.to_numeric(self.movies_df["vote_count"], errors='coerce') >= self.m
        ].copy()
        
        # 计算加权评分
        self.qualified_movies["demographic_score"] = self.qualified_movies.apply(
            self._weighted_rating, axis=1
        )
        
        # 按分数排序
        self.qualified_movies = self.qualified_movies.sort_values(
            'demographic_score', ascending=False
        )
        
        logger.info(f"筛选出 {len(self.qualified_movies)} 部符合条件的电影")
        
    def get_top_movies(self, n: int = 10) -> pd.DataFrame:
        """获取评分最高的N部电影"""
        if self.qualified_movies is None:
            raise ValueError("请先调用 fit() 方法训练模型")
            
        return self.qualified_movies[
            ["title", "vote_count", "vote_average", "demographic_score", "genres"]
        ].head(n)
    
    def get_popular_movies(self, n: int = 10) -> pd.DataFrame:
        """获取最受欢迎的N部电影（基于流行度）"""
        if self.movies_df is None:
            raise ValueError("请先调用 fit() 方法训练模型")
            
        popular_movies = self.movies_df.sort_values("popularity", ascending=False)
        return popular_movies[["title", "popularity", "vote_average", "genres"]].head(n)
    
    def get_movies_by_genre(self, genre: str, n: int = 10) -> pd.DataFrame:
        """根据类型获取推荐电影"""
        if self.qualified_movies is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        # 筛选包含指定类型的电影
        genre_movies = self.qualified_movies[
            self.qualified_movies["genres"].apply(
                lambda x: genre.lower() in [g.lower() for g in x] if isinstance(x, list) else False
            )
        ]
        
        return genre_movies[
            ["title", "vote_count", "vote_average", "demographic_score", "genres"]
        ].head(n)
    
    def get_movies_by_year_range(self, start_year: int, end_year: int, n: int = 10) -> pd.DataFrame:
        """根据年份范围获取推荐电影"""
        if self.qualified_movies is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        # 提取年份
        movies_with_year = self.qualified_movies.copy()
        movies_with_year['release_year'] = pd.to_datetime(
            movies_with_year['release_date']
        ).dt.year
        
        # 筛选年份范围
        year_movies = movies_with_year[
            (movies_with_year['release_year'] >= start_year) &
            (movies_with_year['release_year'] <= end_year)
        ]
        
        return year_movies[
            ["title", "release_year", "vote_count", "vote_average", "demographic_score"]
        ].head(n)
    
    def plot_top_movies(self, n: int = 10):
        """绘制最受欢迎的电影图表"""
        if self.movies_df is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        popular_movies = self.get_popular_movies(n)
        
        plt.figure(figsize=(12, 6))
        plt.barh(
            popular_movies["title"], 
            popular_movies["popularity"], 
            align="center", 
            color="skyblue"
        )
        plt.gca().invert_yaxis()
        plt.title(f"Top {n} 最受欢迎的电影")
        plt.xlabel("流行度")
        plt.tight_layout()
        plt.show()
    
    def plot_rating_distribution(self):
        """绘制评分分布图"""
        if self.movies_df is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.movies_df["vote_average"], bins=20, alpha=0.7, color='skyblue')
        plt.title("电影评分分布")
        plt.xlabel("平均评分")
        plt.ylabel("电影数量")
        
        plt.subplot(1, 2, 2)
        plt.hist(self.movies_df["vote_count"], bins=50, alpha=0.7, color='lightcoral')
        plt.title("投票数分布")
        plt.xlabel("投票数")
        plt.ylabel("电影数量")
        plt.xlim(0, 2000)  # 限制x轴范围以更好地显示分布
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> Dict:
        """获取推荐系统统计信息"""
        if self.movies_df is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        stats = {
            "total_movies": len(self.movies_df),
            "qualified_movies": len(self.qualified_movies) if self.qualified_movies is not None else 0,
            "average_rating": self.C,
            "vote_threshold": self.m,
            "min_rating": self.movies_df["vote_average"].min(),
            "max_rating": self.movies_df["vote_average"].max(),
            "total_votes": self.movies_df["vote_count"].sum()
        }
        
        return stats
    
    def _weighted_rating(self, row) -> float:
        """计算IMDB加权评分"""
        v = row["vote_count"]
        R = row["vote_average"]
        return (v/(v + self.m) * R) + (self.m/(v + self.m) * self.C)
