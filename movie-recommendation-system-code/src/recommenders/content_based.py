"""
基于内容的推荐模块
使用TF-IDF和余弦相似度进行电影推荐
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
import sys
import os

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from config import Config

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """基于内容的推荐器"""
    
    def __init__(self, method: str = 'tfidf'):
        """
        初始化推荐器
        
        Args:
            method: 'tfidf' 或 'count' - 选择向量化方法
        """
        self.method = method
        self.movies_df = None
        self.similarity_matrix = None
        self.indices = None
        self.vectorizer = None
        
        # 选择向量化器
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        elif method == 'count':
            self.vectorizer = CountVectorizer(stop_words="english", max_features=5000)
        else:
            raise ValueError("method 必须是 'tfidf' 或 'count'")
    
    def fit(self, movies_df: pd.DataFrame, content_column: str = 'overview'):
        """
        训练基于内容的推荐器
        
        Args:
            movies_df: 电影数据框
            content_column: 用于计算相似度的内容列
        """
        logger.info(f"使用 {self.method} 方法训练基于内容的推荐器...")
        
        self.movies_df = movies_df.copy()
        self.content_column = content_column
        
        # 确保内容列存在且处理缺失值
        if content_column not in self.movies_df.columns:
            raise ValueError(f"列 '{content_column}' 不存在于数据中")
        
        self.movies_df[content_column] = self.movies_df[content_column].fillna("")
        
        # 向量化文本内容
        text_matrix = self.vectorizer.fit_transform(self.movies_df[content_column])
        
        # 计算相似度矩阵
        if self.method == 'tfidf':
            self.similarity_matrix = linear_kernel(text_matrix, text_matrix)
        else:
            self.similarity_matrix = cosine_similarity(text_matrix, text_matrix)
        
        # 创建标题到索引的映射
        self.indices = pd.Series(
            self.movies_df.index, 
            index=self.movies_df["title"]
        ).drop_duplicates()
        
        logger.info(f"相似度矩阵计算完成，形状: {self.similarity_matrix.shape}")
    
    def get_recommendations(self, title: str, n: int = 10) -> List[Dict]:
        """
        获取基于内容的推荐
        
        Args:
            title: 电影标题
            n: 推荐数量
            
        Returns:
            推荐电影列表
        """
        if self.similarity_matrix is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        if title not in self.indices:
            available_titles = self.indices.index.tolist()[:10]
            raise ValueError(f"电影 '{title}' 不存在。可用的电影包括: {available_titles}")
        
        # 获取电影索引
        idx = self.indices[title]
        
        # 计算相似度分数
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # 排除自身，取前n个
        sim_scores = sim_scores[1:n+1]
        
        # 获取推荐电影信息
        movie_indices = [i[0] for i in sim_scores]
        recommendations = []
        
        for i, movie_idx in enumerate(movie_indices):
            movie_data = self.movies_df.iloc[movie_idx]
            recommendations.append({
                'rank': i + 1,
                'title': movie_data['title'],
                'similarity_score': sim_scores[i][1],
                'vote_average': movie_data.get('vote_average', 'N/A'),
                'release_date': movie_data.get('release_date', 'N/A'),
                'genres': movie_data.get('genres', []),
                'overview': movie_data.get('overview', '')[:100] + '...' if movie_data.get('overview') else 'N/A'
            })
        
        return recommendations
    
    def get_similar_movies_by_features(self, title: str, feature_weights: Dict[str, float] = None, n: int = 10) -> List[Dict]:
        """
        基于多种特征的相似电影推荐
        
        Args:
            title: 电影标题
            feature_weights: 特征权重字典，如 {'genres': 0.4, 'director': 0.3, 'cast': 0.3}
            n: 推荐数量
        """
        if title not in self.indices:
            raise ValueError(f"电影 '{title}' 不存在")
        
        if feature_weights is None:
            feature_weights = {'genres': 0.4, 'director': 0.3, 'cast': 0.3}
        
        target_idx = self.indices[title]
        target_movie = self.movies_df.iloc[target_idx]
        
        # 计算基于特征的相似度
        similarities = []
        
        for idx, movie in self.movies_df.iterrows():
            if idx == target_idx:
                continue
                
            total_similarity = 0
            
            # 计算各种特征的相似度
            for feature, weight in feature_weights.items():
                if feature in self.movies_df.columns:
                    similarity = self._calculate_feature_similarity(
                        target_movie.get(feature, []), 
                        movie.get(feature, [])
                    )
                    total_similarity += similarity * weight
            
            similarities.append((idx, total_similarity))
        
        # 排序并获取top-n
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:n]
        
        # 构建推荐结果
        recommendations = []
        for i, (movie_idx, sim_score) in enumerate(top_similarities):
            movie_data = self.movies_df.iloc[movie_idx]
            recommendations.append({
                'rank': i + 1,
                'title': movie_data['title'],
                'feature_similarity': sim_score,
                'vote_average': movie_data.get('vote_average', 'N/A'),
                'release_date': movie_data.get('release_date', 'N/A'),
                'genres': movie_data.get('genres', []),
                'director': movie_data.get('director', 'N/A'),
                'cast': movie_data.get('cast', [])
            })
        
        return recommendations
    
    def find_similar_by_genre(self, genres: List[str], n: int = 10) -> List[Dict]:
        """
        根据类型查找相似电影
        
        Args:
            genres: 类型列表
            n: 推荐数量
        """
        if self.movies_df is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        # 筛选包含指定类型的电影
        matching_movies = []
        
        for idx, movie in self.movies_df.iterrows():
            movie_genres = movie.get('genres', [])
            if isinstance(movie_genres, list):
                # 计算类型匹配度
                genre_overlap = len(set(genres) & set(movie_genres))
                if genre_overlap > 0:
                    matching_movies.append({
                        'title': movie['title'],
                        'genres': movie_genres,
                        'genre_match_score': genre_overlap / len(genres),
                        'vote_average': movie.get('vote_average', 'N/A'),
                        'release_date': movie.get('release_date', 'N/A')
                    })
        
        # 按匹配度和评分排序
        matching_movies.sort(
            key=lambda x: (x['genre_match_score'], x.get('vote_average', 0)), 
            reverse=True
        )
        
        return matching_movies[:n]
    
    def get_content_statistics(self) -> Dict:
        """获取内容统计信息"""
        if self.movies_df is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        stats = {
            'total_movies': len(self.movies_df),
            'vectorization_method': self.method,
            'similarity_matrix_shape': self.similarity_matrix.shape if self.similarity_matrix is not None else None,
            'content_column': self.content_column,
            'average_similarity': np.mean(self.similarity_matrix) if self.similarity_matrix is not None else None,
            'max_similarity': np.max(self.similarity_matrix) if self.similarity_matrix is not None else None
        }
        
        return stats
    
    def _calculate_feature_similarity(self, feature1, feature2) -> float:
        """计算两个特征之间的相似度"""
        if not feature1 or not feature2:
            return 0.0
        
        # 转换为集合
        if isinstance(feature1, str):
            feature1 = [feature1]
        if isinstance(feature2, str):
            feature2 = [feature2]
        
        set1 = set(feature1) if isinstance(feature1, list) else set()
        set2 = set(feature2) if isinstance(feature2, list) else set()
        
        if not set1 or not set2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_movie_titles(self) -> List[str]:
        """获取所有可用的电影标题"""
        if self.indices is None:
            return []
        return self.indices.index.tolist()
