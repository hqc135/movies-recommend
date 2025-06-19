"""
推荐算法测试
"""
import pytest
import pandas as pd
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.recommenders.demographic import DemographicRecommender
from src.recommenders.content_based import ContentBasedRecommender

class TestDemographicRecommender:
    """人口统计学推荐器测试"""
    
    def test_init(self):
        """测试初始化"""
        recommender = DemographicRecommender()
        assert recommender is not None
    
    def test_recommend(self, sample_data):
        """测试推荐功能"""
        recommender = DemographicRecommender()
        # 需要实际的数据来测试
        # recommendations = recommender.recommend(count=5)
        # assert len(recommendations) <= 5

class TestContentBasedRecommender:
    """基于内容推荐器测试"""
    
    def test_init(self):
        """测试初始化"""
        recommender = ContentBasedRecommender('tfidf')
        assert recommender.method == 'tfidf'
    
    def test_recommend_similar(self, sample_data):
        """测试相似电影推荐"""
        recommender = ContentBasedRecommender('tfidf')
        # 需要实际的数据来测试
        # recommendations = recommender.recommend_similar('Test Movie 1', count=5)
        # assert len(recommendations) <= 5
