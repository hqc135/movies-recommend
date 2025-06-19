"""
电影推荐系统 - Web应用
基于Flask的现代化Web界面
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import sys
import json
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config

from src.data.data_loader import DataLoader
from src.recommenders.demographic import DemographicRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.utils.utils import setup_logger

try:
    from src.recommenders.deep_learning import DeepLearningRecommender
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'movie_recommender_secret_key_2025'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = setup_logger("WebApp")

# 全局变量存储推荐系统
recommendation_system = None
movies_data = None

class WebMovieRecommendationSystem:
    """Web版电影推荐系统"""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger("WebMovieRecommendationSystem")
        self.is_initialized = False
        
        # 初始化组件
        self.data_loader = DataLoader(self.config)
        self.demographic_recommender = DemographicRecommender(self.config)
        self.content_recommender = ContentBasedRecommender('tfidf')
        self.hybrid_recommender = HybridRecommender(self.config)
        
        # 深度学习推荐器（如果可用）
        self.deep_learning_recommender = None
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.deep_learning_recommender = DeepLearningRecommender(
                    embedding_size=self.config.EMBEDDING_DIM,
                    hidden_units=self.config.NEURAL_LAYERS,
                    learning_rate=self.config.LEARNING_RATE,
                    dropout_rate=self.config.DROPOUT_RATE
                )
            except Exception as e:
                self.logger.warning(f"深度学习推荐器初始化失败: {e}")
        
        self.movies_df = None
        
    def initialize(self):
        """初始化系统（加载数据和训练模型）"""
        if self.is_initialized:
            return True
            
        try:
            self.logger.info("开始初始化推荐系统...")
            
            # 加载数据
            self.movies_df = self.data_loader.get_processed_data()
            self.logger.info(f"数据加载完成: {len(self.movies_df)} 部电影")
            
            # 训练推荐器
            self.demographic_recommender.fit(self.movies_df)
            self.content_recommender.fit(self.movies_df)
            
            # 初始化混合推荐器
            recommenders = {
                'demographic': self.demographic_recommender,
                'content_based': self.content_recommender
            }
            
            if self.deep_learning_recommender:
                self.deep_learning_recommender.fit(self.movies_df)
                recommenders['deep_learning'] = self.deep_learning_recommender
            
            self.hybrid_recommender.fit(self.movies_df, recommenders)
            
            self.is_initialized = True
            self.logger.info("推荐系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def get_top_movies(self, n=20):
        """获取评分最高的电影"""
        if not self.is_initialized:
            return []
        
        try:
            recommendations = self.demographic_recommender.recommend(n=n)
            return self._format_recommendations(recommendations)
        except Exception as e:
            self.logger.error(f"获取热门电影失败: {e}")
            return []
    
    def get_content_recommendations(self, movie_title, n=10):
        """获取基于内容的推荐"""
        if not self.is_initialized:
            return []
        
        try:
            recommendations = self.content_recommender.recommend(movie_title, n=n)
            return self._format_recommendations(recommendations)
        except Exception as e:
            self.logger.error(f"内容推荐失败: {e}")
            return []
    
    def get_hybrid_recommendations(self, movie_title=None, n=10):
        """获取混合推荐"""
        if not self.is_initialized:
            return []
        
        try:
            if movie_title:
                recommendations = self.hybrid_recommender.recommend(movie_title, n=n)
            else:
                # 如果没有指定电影，返回热门推荐
                recommendations = self.demographic_recommender.recommend(n=n)
            return self._format_recommendations(recommendations)
        except Exception as e:
            self.logger.error(f"混合推荐失败: {e}")
            return []
    
    def search_movies(self, query, limit=20):
        """搜索电影"""
        if not self.is_initialized or self.movies_df is None:
            return []
        
        try:
            # 按标题搜索
            mask = self.movies_df['title'].str.contains(query, case=False, na=False)
            results = self.movies_df[mask].head(limit)
            
            return [{
                'title': row['title'],
                'overview': row.get('overview', '')[:200] + '...' if len(row.get('overview', '')) > 200 else row.get('overview', ''),
                'rating': round(row.get('vote_average', 0), 1),
                'year': str(row.get('release_date', ''))[:4] if pd.notna(row.get('release_date')) else 'N/A',
                'genres': ', '.join(row.get('genres', [])) if isinstance(row.get('genres'), list) else str(row.get('genres', ''))
            } for _, row in results.iterrows()]
            
        except Exception as e:
            self.logger.error(f"搜索电影失败: {e}")
            return []
    
    def get_movie_details(self, movie_title):
        """获取电影详细信息"""
        if not self.is_initialized or self.movies_df is None:
            return None
        
        try:
            movie = self.movies_df[self.movies_df['title'] == movie_title]
            if movie.empty:
                return None
            
            movie = movie.iloc[0]
            return {
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'rating': round(movie.get('vote_average', 0), 1),
                'vote_count': int(movie.get('vote_count', 0)),
                'year': str(movie.get('release_date', ''))[:4] if pd.notna(movie.get('release_date')) else 'N/A',
                'genres': ', '.join(movie.get('genres', [])) if isinstance(movie.get('genres'), list) else str(movie.get('genres', '')),
                'director': movie.get('director', 'N/A'),
                'cast': ', '.join(movie.get('cast', [])) if isinstance(movie.get('cast'), list) else str(movie.get('cast', '')),
                'budget': f"${movie.get('budget', 0):,}" if movie.get('budget', 0) > 0 else 'N/A',
                'revenue': f"${movie.get('revenue', 0):,}" if movie.get('revenue', 0) > 0 else 'N/A'
            }
        except Exception as e:
            self.logger.error(f"获取电影详情失败: {e}")
            return None
    
    def _format_recommendations(self, recommendations):
        """格式化推荐结果"""
        formatted = []
        for rec in recommendations:
            formatted.append({
                'title': rec.get('title', 'Unknown'),
                'score': round(rec.get('hybrid_score', rec.get('weighted_score', rec.get('similarity_score', 0))), 3),
                'rating': round(rec.get('vote_average', 0), 1),
                'year': str(rec.get('release_date', ''))[:4] if pd.notna(rec.get('release_date')) else 'N/A',
                'genres': ', '.join(rec.get('genres', [])) if isinstance(rec.get('genres'), list) else str(rec.get('genres', ''))
            })
        return formatted

# 路由定义
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/initialize')
def initialize_system():
    """初始化推荐系统"""
    global recommendation_system
    
    if recommendation_system is None:
        recommendation_system = WebMovieRecommendationSystem()
    
    success = recommendation_system.initialize()
    
    if success:
        return jsonify({'status': 'success', 'message': '推荐系统初始化成功'})
    else:
        return jsonify({'status': 'error', 'message': '推荐系统初始化失败'}), 500

@app.route('/top-movies')
def top_movies():
    """获取热门电影"""
    global recommendation_system
    
    if recommendation_system is None or not recommendation_system.is_initialized:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    movies = recommendation_system.get_top_movies(n=20)
    return jsonify(movies)

@app.route('/search')
def search():
    """搜索电影"""
    global recommendation_system
    
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    
    if recommendation_system is None or not recommendation_system.is_initialized:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    results = recommendation_system.search_movies(query)
    return jsonify(results)

@app.route('/movie/<movie_title>')
def movie_detail(movie_title):
    """电影详情页"""
    global recommendation_system
    
    if recommendation_system is None or not recommendation_system.is_initialized:
        return redirect(url_for('index'))
    
    movie = recommendation_system.get_movie_details(movie_title)
    if movie is None:
        return render_template('404.html'), 404
    
    # 获取相似电影推荐
    similar_movies = recommendation_system.get_content_recommendations(movie_title, n=6)
    
    return render_template('movie_detail.html', movie=movie, similar_movies=similar_movies)

@app.route('/recommend')
def recommend():
    """推荐页面"""
    global recommendation_system
    
    movie_title = request.args.get('movie')
    recommend_type = request.args.get('type', 'hybrid')
    
    if recommendation_system is None or not recommendation_system.is_initialized:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    try:
        if recommend_type == 'content' and movie_title:
            recommendations = recommendation_system.get_content_recommendations(movie_title, n=10)
        elif recommend_type == 'hybrid':
            recommendations = recommendation_system.get_hybrid_recommendations(movie_title, n=10)
        else:
            recommendations = recommendation_system.get_top_movies(n=10)
        
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"推荐失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """系统状态"""
    global recommendation_system
    
    if recommendation_system is None:
        return jsonify({'initialized': False, 'deep_learning_available': DEEP_LEARNING_AVAILABLE})
    
    return jsonify({
        'initialized': recommendation_system.is_initialized,
        'deep_learning_available': DEEP_LEARNING_AVAILABLE,
        'total_movies': len(recommendation_system.movies_df) if recommendation_system.movies_df is not None else 0
    })

if __name__ == '__main__':
    print("🎬 启动电影推荐系统Web界面...")
    print("📍 访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
