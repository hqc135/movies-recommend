"""
混合推荐模块
结合多种推荐算法的混合推荐系统
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HybridRecommender:
    """混合推荐系统"""
    
    def __init__(self, config=None):
        self.config = config
        self.recommenders = {}
        self.weights = {}
        self.is_fitted = False
        
    def add_recommender(self, name: str, recommender, weight: float = 1.0):
        """
        添加推荐器
        
        Args:
            name: 推荐器名称
            recommender: 推荐器实例
            weight: 权重
        """
        self.recommenders[name] = recommender
        self.weights[name] = weight
        logger.info(f"添加推荐器: {name}, 权重: {weight}")
    
    def fit(self, movies_df: pd.DataFrame):
        """训练所有推荐器"""
        logger.info("训练混合推荐系统...")
        
        for name, recommender in self.recommenders.items():
            logger.info(f"训练推荐器: {name}")
            try:
                if hasattr(recommender, 'fit'):
                    recommender.fit(movies_df)
                else:
                    logger.warning(f"推荐器 {name} 没有fit方法")
            except Exception as e:
                logger.error(f"训练推荐器 {name} 时出错: {e}")
        
        self.is_fitted = True
        logger.info("混合推荐系统训练完成")
    
    def recommend(self, **kwargs) -> List[Dict]:
        """
        生成混合推荐
        
        Args:
            **kwargs: 传递给各个推荐器的参数
            
        Returns:
            混合推荐结果
        """
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        all_recommendations = {}
        
        # 收集各个推荐器的结果
        for name, recommender in self.recommenders.items():
            try:
                if name == 'content_based' and hasattr(recommender, 'get_recommendations'):
                    # 基于内容的推荐
                    title = kwargs.get('title', '')
                    if title:
                        recs = recommender.get_recommendations(title, kwargs.get('n', 10))
                        all_recommendations[name] = self._process_content_recommendations(recs)
                
                elif name == 'demographic' and hasattr(recommender, 'get_top_movies'):
                    # 人口统计学推荐
                    recs = recommender.get_top_movies(kwargs.get('n', 10))
                    all_recommendations[name] = self._process_demographic_recommendations(recs)
                
                elif name == 'deep_learning' and hasattr(recommender, 'recommend_movies'):
                    # 深度学习推荐
                    user_id = kwargs.get('user_id', 'default_user')
                    movies_df = kwargs.get('movies_df')
                    if movies_df is not None:
                        recs = recommender.recommend_movies(user_id, movies_df, kwargs.get('n', 10))
                        all_recommendations[name] = recs
                
            except Exception as e:
                logger.warning(f"推荐器 {name} 生成推荐时出错: {e}")
                continue
        
        # 合并推荐结果
        hybrid_recommendations = self._merge_recommendations(all_recommendations, kwargs.get('n', 10))
        
        return hybrid_recommendations
    
    def _process_content_recommendations(self, recommendations) -> List[Dict]:
        """处理基于内容的推荐结果"""
        processed = []
        
        if isinstance(recommendations, pd.Series):
            # 如果是pandas Series，转换为字典列表
            for i, title in enumerate(recommendations):
                processed.append({
                    'title': title,
                    'content_score': 1.0 - (i * 0.1),  # 基于排名的分数
                    'rank': i + 1
                })
        elif isinstance(recommendations, list):
            # 如果已经是字典列表
            for i, rec in enumerate(recommendations):
                if isinstance(rec, dict):
                    processed.append({
                        'title': rec.get('title', ''),
                        'content_score': rec.get('similarity_score', 1.0 - (i * 0.1)),
                        'rank': i + 1
                    })
                else:
                    processed.append({
                        'title': str(rec),
                        'content_score': 1.0 - (i * 0.1),
                        'rank': i + 1
                    })
        
        return processed
    
    def _process_demographic_recommendations(self, recommendations) -> List[Dict]:
        """处理人口统计学推荐结果"""
        processed = []
        
        if isinstance(recommendations, pd.DataFrame):
            for i, (_, row) in enumerate(recommendations.iterrows()):
                processed.append({
                    'title': row.get('title', ''),
                    'demographic_score': row.get('demographic_score', row.get('weighted_score', 0.5)),
                    'vote_average': row.get('vote_average', 0),
                    'rank': i + 1
                })
        
        return processed
    
    def _merge_recommendations(self, all_recommendations: Dict[str, List[Dict]], n: int) -> List[Dict]:
        """合并所有推荐结果"""
        # 收集所有电影标题
        all_movies = {}
        
        # 为每个电影累计分数
        for recommender_name, recommendations in all_recommendations.items():
            weight = self.weights.get(recommender_name, 1.0)
            
            for rec in recommendations:
                title = rec.get('title', '')
                if not title:
                    continue
                
                if title not in all_movies:
                    all_movies[title] = {
                        'title': title,
                        'total_score': 0.0,
                        'recommender_scores': {},
                        'additional_info': {}
                    }
                
                # 计算标准化分数
                if recommender_name == 'content_based':
                    score = rec.get('content_score', rec.get('similarity_score', 0.5))
                elif recommender_name == 'demographic':
                    score = rec.get('demographic_score', 0.5)
                elif recommender_name == 'deep_learning':
                    score = rec.get('predicted_rating', 5.0) / 10.0  # 标准化到0-1
                else:
                    score = 0.5
                
                # 应用权重并累加
                weighted_score = score * weight
                all_movies[title]['total_score'] += weighted_score
                all_movies[title]['recommender_scores'][recommender_name] = score
                
                # 保存额外信息
                for key, value in rec.items():
                    if key not in ['title', 'rank']:
                        all_movies[title]['additional_info'][key] = value
        
        # 排序并返回top-n
        sorted_movies = sorted(
            all_movies.values(), 
            key=lambda x: x['total_score'], 
            reverse=True
        )
        
        # 格式化最终结果
        final_recommendations = []
        for i, movie in enumerate(sorted_movies[:n]):
            final_rec = {
                'rank': i + 1,
                'title': movie['title'],
                'hybrid_score': movie['total_score'],
                'recommender_scores': movie['recommender_scores']
            }
            final_rec.update(movie['additional_info'])
            final_recommendations.append(final_rec)
        
        return final_recommendations
    
    def get_recommender_contributions(self, recommendations: List[Dict]) -> Dict[str, float]:
        """分析各推荐器的贡献度"""
        contributions = {}
        total_weight = sum(self.weights.values())
        
        for name, weight in self.weights.items():
            contributions[name] = weight / total_weight if total_weight > 0 else 0
        
        return contributions
    
    def set_weights(self, weights: Dict[str, float]):
        """设置推荐器权重"""
        for name, weight in weights.items():
            if name in self.recommenders:
                self.weights[name] = weight
                logger.info(f"更新推荐器 {name} 权重为: {weight}")
            else:
                logger.warning(f"推荐器 {name} 不存在")
    
    def get_statistics(self) -> Dict:
        """获取混合推荐系统统计信息"""
        stats = {
            'num_recommenders': len(self.recommenders),
            'recommender_names': list(self.recommenders.keys()),
            'weights': self.weights.copy(),
            'is_fitted': self.is_fitted,
            'total_weight': sum(self.weights.values())
        }
        
        return stats
    
    def explain_recommendation(self, title: str, recommendations: List[Dict]) -> Dict:
        """解释推荐结果"""
        explanation = {
            'target_movie': title,
            'total_recommendations': len(recommendations),
            'recommender_contributions': self.get_recommender_contributions(recommendations)
        }
        
        # 找到目标电影的推荐详情
        target_rec = None
        for rec in recommendations:
            if rec.get('title') == title:
                target_rec = rec
                break
        
        if target_rec:
            explanation['recommendation_details'] = {
                'hybrid_score': target_rec.get('hybrid_score', 0),
                'individual_scores': target_rec.get('recommender_scores', {}),
                'rank': target_rec.get('rank', 'N/A')
            }
        
        return explanation
