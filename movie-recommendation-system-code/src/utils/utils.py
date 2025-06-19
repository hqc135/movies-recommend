"""
工具模块
包含日志设置、可视化、评估等工具函数
"""
import logging
import sys
import os

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from config import Config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import seaborn as sns

def setup_logger(name: str = "movie_recommender", level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # 创建控制台处理器
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # 添加处理器到日志器
        logger.addHandler(handler)
    
    return logger

class RecommendationVisualizer:
    """推荐系统可视化工具"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """初始化可视化器"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
          # 设置中文字体支持
        self._setup_chinese_font()
        
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def _setup_chinese_font(self):
        """设置中文字体支持"""
        try:
            import matplotlib.font_manager as fm
            
            # Windows常见中文字体
            windows_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
            
            # 获取系统所有字体
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # 寻找可用的中文字体
            chinese_font = None
            for font in windows_fonts:
                if font in available_fonts:
                    chinese_font = font
                    break
            
            if chinese_font:
                plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 已设置中文字体: {chinese_font}")
            else:
                # 使用通用设置
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print("⚠ 未找到中文字体，使用默认字体，中文可能显示为方框")
                
        except ImportError:
            # 如果无法导入font_manager，使用基本设置
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠ 使用基本字体设置")
    
    def plot_recommendation_scores(self, recommendations: List[Dict], 
                                 score_column: str = 'hybrid_score',
                                 title: str = "推荐评分分布"):
        """绘制推荐评分分布"""
        if not recommendations:
            print("没有推荐数据可显示")
            return
        
        # 提取数据
        titles = [rec.get('title', f'Movie {i}') for i, rec in enumerate(recommendations)]
        scores = [rec.get(score_column, 0) for rec in recommendations]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(titles)), scores, color=self.colors[0], alpha=0.7)
        
        # 设置标签
        plt.yticks(range(len(titles)), titles)
        plt.xlabel('评分')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_recommender_comparison(self, recommendations: List[Dict], 
                                  recommender_names: List[str]):
        """比较不同推荐器的分数"""
        if not recommendations:
            print("没有推荐数据可显示")
            return
        
        # 准备数据
        data = []
        for rec in recommendations[:10]:  # 只显示前10个
            title = rec.get('title', 'Unknown')
            scores = rec.get('recommender_scores', {})
            for recommender in recommender_names:
                if recommender in scores:
                    data.append({
                        'title': title,
                        'recommender': recommender,
                        'score': scores[recommender]
                    })
        
        if not data:
            print("没有推荐器分数数据可显示")
            return
        
        # 创建DataFrame并绘图
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        
        # 创建分组柱状图
        titles = df['title'].unique()
        x = np.arange(len(titles))
        width = 0.8 / len(recommender_names)
        
        for i, recommender in enumerate(recommender_names):
            recommender_data = df[df['recommender'] == recommender]
            scores = [recommender_data[recommender_data['title'] == title]['score'].iloc[0] 
                     if len(recommender_data[recommender_data['title'] == title]) > 0 else 0 
                     for title in titles]
            
            plt.bar(x + i * width, scores, width, 
                   label=recommender, color=self.colors[i % len(self.colors)], alpha=0.8)
        
        plt.xlabel('电影')
        plt.ylabel('分数')
        plt.title('不同推荐器分数比较')
        plt.xticks(x + width * (len(recommender_names) - 1) / 2, 
                  [str(title) for title in titles], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_genre_distribution(self, movies_df: pd.DataFrame):
        """绘制类型分布图"""
        if 'genres' not in movies_df.columns:
            print("数据中没有genres列")
            return
        
        # 统计类型
        genre_counts = {}
        for _, row in movies_df.iterrows():
            genres = row.get('genres', [])
            if isinstance(genres, list):
                for genre in genres:
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if not genre_counts:
            print("没有类型数据")
            return
        
        # 排序并选择前15个
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        genres, counts = zip(*sorted_genres)
        
        # 绘制图表
        plt.figure(figsize=(12, 6))
        plt.bar(genres, counts, color=self.colors[2], alpha=0.7)
        plt.xlabel('电影类型')
        plt.ylabel('数量')
        plt.title('电影类型分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_rating_distribution(self, movies_df: pd.DataFrame):
        """绘制评分分布"""
        if 'vote_average' not in movies_df.columns:
            print("数据中没有vote_average列")
            return
        
        plt.figure(figsize=(12, 4))
        
        # 评分分布
        plt.subplot(1, 2, 1)
        plt.hist(movies_df['vote_average'].dropna(), bins=20, 
                color=self.colors[1], alpha=0.7, edgecolor='black')
        plt.xlabel('平均评分')
        plt.ylabel('电影数量')
        plt.title('电影评分分布')
        plt.grid(True, alpha=0.3)
        
        # 投票数分布
        if 'vote_count' in movies_df.columns:
            plt.subplot(1, 2, 2)
            plt.hist(movies_df['vote_count'].dropna(), bins=50, 
                    color=self.colors[3], alpha=0.7, edgecolor='black')
            plt.xlabel('投票数')
            plt.ylabel('电影数量')
            plt.title('投票数分布')
            plt.xlim(0, movies_df['vote_count'].quantile(0.95))  # 限制x轴以更好显示
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class RecommendationEvaluator:
    """推荐系统评估工具"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_diversity(self, recommendations: List[Dict]) -> float:
        """
        计算推荐多样性
        
        Args:
            recommendations: 推荐结果列表
            
        Returns:
            多样性分数
        """
        if not recommendations:
            return 0.0
        
        # 基于类型的多样性
        all_genres = set()
        movie_genres = []
        
        for rec in recommendations:
            genres = rec.get('genres', [])
            if isinstance(genres, list):
                movie_genres.append(set(genres))
                all_genres.update(genres)
        
        if not all_genres:
            return 0.0
        
        # 计算平均类型覆盖率
        diversity_scores = []
        for genres in movie_genres:
            if all_genres:
                diversity_scores.append(len(genres) / len(all_genres))
        
        return float(np.mean(diversity_scores)) if diversity_scores else 0.0
    
    def calculate_coverage(self, recommendations: List[Dict], 
                         total_items: int) -> float:
        """
        计算推荐覆盖率
        
        Args:
            recommendations: 推荐结果列表
            total_items: 总项目数
            
        Returns:
            覆盖率
        """
        if total_items == 0:
            return 0.0
        
        unique_items = len(set(rec.get('title', '') for rec in recommendations))
        return unique_items / total_items
    
    def calculate_novelty(self, recommendations: List[Dict]) -> float:
        """
        计算推荐新颖性（基于流行度的倒数）
        
        Args:
            recommendations: 推荐结果列表
            
        Returns:
            新颖性分数
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for rec in recommendations:
            popularity = rec.get('popularity', 1)
            vote_count = rec.get('vote_count', 1)
            
            # 流行度越低，新颖性越高
            novelty = 1.0 / (1.0 + np.log(max(popularity, 1)))
            novelty_scores.append(novelty)
        
        return float(np.mean(novelty_scores))
    
    def generate_evaluation_report(self, recommendations: List[Dict], 
                                 total_items: Optional[int] = None) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            recommendations: 推荐结果列表
            total_items: 总项目数
            
        Returns:
            评估报告
        """
        report = {
            'total_recommendations': len(recommendations),
            'diversity': self.calculate_diversity(recommendations),
            'novelty': self.calculate_novelty(recommendations)
        }
        
        if total_items:
            report['coverage'] = self.calculate_coverage(recommendations, total_items)
        
        # 评分统计
        scores = [rec.get('hybrid_score', rec.get('predicted_rating', 0)) 
                 for rec in recommendations]
        if scores:
            report['score_statistics'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return report

def print_recommendations(recommendations: List[Dict], title: str = "推荐结果"):
    """格式化打印推荐结果"""
    print(f"\n{'='*20} {title} {'='*20}")
    
    if not recommendations:
        print("没有推荐结果")
        return
    
    for i, rec in enumerate(recommendations, 1):
        movie_title = rec.get('title', f'Movie {i}')
        score = rec.get('hybrid_score', rec.get('predicted_rating', rec.get('similarity_score', 0)))
        
        print(f"{i:2d}. {movie_title:<30} 分数: {score:.3f}")
        
        # 显示额外信息
        if 'vote_average' in rec:
            print(f"    评分: {rec['vote_average']:.1f}")
        if 'genres' in rec and isinstance(rec['genres'], list):
            genres_str = ', '.join(rec['genres'][:3])  # 只显示前3个类型
            print(f"    类型: {genres_str}")
        if 'release_date' in rec:
            print(f"    上映日期: {rec['release_date']}")
        print()

def format_time_duration(seconds: float) -> str:
    """格式化时间长度"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
