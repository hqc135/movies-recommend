"""
电影推荐系统主程序
整合所有模块，提供统一的接口
"""
import sys
import os
import time
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.data.data_loader import DataLoader
from src.recommenders.demographic import DemographicRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.utils.utils import (
    setup_logger, RecommendationVisualizer, RecommendationEvaluator,
    print_recommendations, format_time_duration
)

# 尝试导入深度学习模块
try:
    from src.recommenders.deep_learning import DeepLearningRecommender
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"深度学习模块不可用: {e}")

class MovieRecommendationSystem:
    """电影推荐系统主类"""
    
    def __init__(self, config=None):
        """初始化推荐系统"""
        self.config = config or Config()
        self.logger = setup_logger("MovieRecommendationSystem")
        
        # 初始化组件
        self.data_loader = DataLoader(self.config)
        self.demographic_recommender = DemographicRecommender(self.config)
        self.content_recommender = ContentBasedRecommender('tfidf')
        self.content_metadata_recommender = ContentBasedRecommender('count')
        self.hybrid_recommender = HybridRecommender(self.config)
        
        # 可视化和评估工具
        self.visualizer = RecommendationVisualizer()
        self.evaluator = RecommendationEvaluator()
        
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
        
        # 数据
        self.movies_df = None
        self.is_fitted = False
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        self.logger.info("开始加载和预处理数据...")
        start_time = time.time()
        
        try:
            # 加载数据
            self.movies_df = self.data_loader.get_processed_data()
            
            # 显示数据摘要
            summary = self.data_loader.get_data_summary()
            self.logger.info(f"数据加载完成: {summary}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"数据预处理耗时: {format_time_duration(elapsed_time)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            return False
    
    def fit_all_recommenders(self):
        """训练所有推荐器"""
        if self.movies_df is None:
            raise ValueError("请先加载数据")            
        self.logger.info("开始训练所有推荐器...")
        start_time = time.time()
        
        try:
            # 训练人口统计学推荐器
            self.demographic_recommender.fit(self.movies_df)
            
            # 训练基于内容的推荐器（使用overview）
            self.content_recommender.fit(self.movies_df, 'overview')
            
            # 训练基于元数据的推荐器（使用soup）
            self.content_metadata_recommender.fit(self.movies_df, 'soup')
            
            # 设置混合推荐器
            self.hybrid_recommender.add_recommender(
                'demographic', self.demographic_recommender, 
                self.config.HYBRID_WEIGHTS.get('demographic', 0.2)
            )
            self.hybrid_recommender.add_recommender(
                'content_based', self.content_recommender,
                self.config.HYBRID_WEIGHTS.get('content_based', 0.4)
            )
            
            # 训练深度学习推荐器（如果可用）
            if self.deep_learning_recommender:
                try:
                    # 准备交互数据
                    interactions_df = self.deep_learning_recommender.prepare_interaction_data(
                        self.movies_df.head(100)  # 使用前100部电影以加快训练
                    )
                    
                    # 构建模型
                    num_users = interactions_df['user_encoded'].nunique()
                    num_movies = interactions_df['movie_encoded'].nunique()
                    self.deep_learning_recommender.build_model(num_users, num_movies)
                    
                    # 训练模型
                    self.deep_learning_recommender.train(
                        interactions_df, 
                        epochs=self.config.EPOCHS,
                        batch_size=self.config.BATCH_SIZE
                    )
                    
                    # 添加到混合推荐器
                    self.hybrid_recommender.add_recommender(
                        'deep_learning', self.deep_learning_recommender,
                        self.config.HYBRID_WEIGHTS.get('deep_learning', 0.6)
                    )
                    
                    self.logger.info("深度学习推荐器训练完成")
                    
                except Exception as e:
                    self.logger.warning(f"深度学习推荐器训练失败: {e}")
            
            # 训练混合推荐器
            self.hybrid_recommender.fit(self.movies_df)
            
            self.is_fitted = True
            elapsed_time = time.time() - start_time
            self.logger.info(f"所有推荐器训练完成，耗时: {format_time_duration(elapsed_time)}")
            
        except Exception as e:
            self.logger.error(f"推荐器训练失败: {e}")
            raise
    
    def get_demographic_recommendations(self, n=10):
        """获取人口统计学推荐"""
        if not self.is_fitted:
            raise ValueError("请先训练推荐器")
        
        return self.demographic_recommender.get_top_movies(n)
    
    def get_content_recommendations(self, title, n=10, use_metadata=False):
        """获取基于内容的推荐"""
        if not self.is_fitted:
            raise ValueError("请先训练推荐器")
        
        recommender = self.content_metadata_recommender if use_metadata else self.content_recommender
        return recommender.get_recommendations(title, n)
    
    def get_hybrid_recommendations(self, title=None, user_id="default_user", n=10):
        """获取混合推荐"""
        if not self.is_fitted:
            raise ValueError("请先训练推荐器")
        
        kwargs = {
            'n': n,
            'movies_df': self.movies_df
        }
        
        if title:
            kwargs['title'] = title
        if user_id:
            kwargs['user_id'] = user_id
            
        return self.hybrid_recommender.recommend(**kwargs)
    
    def demonstrate_system(self):
        """系统演示"""
        if not self.is_fitted:
            raise ValueError("请先训练推荐器")
        
        print("\n" + "="*60)
        print("           电影推荐系统演示")
        print("="*60)
        
        # 1. 人口统计学推荐
        print("\n🏆 人口统计学推荐 - 最受好评的电影:")
        demo_recs = self.get_demographic_recommendations(5)
        for i, (_, movie) in enumerate(demo_recs.iterrows(), 1):
            print(f"{i}. {movie['title']} (评分: {movie['vote_average']:.1f}, 加权分数: {movie['demographic_score']:.3f})")
        
        # 2. 基于内容的推荐
        sample_movies = ["The Dark Knight Rises", "The Avengers", "Inception"]
        available_movies = [movie for movie in sample_movies 
                          if movie in self.content_recommender.get_movie_titles()]
        
        if available_movies:
            sample_movie = available_movies[0]
            print(f"\n🎬 基于内容推荐 - 与《{sample_movie}》相似的电影:")
            content_recs = self.get_content_recommendations(sample_movie, 5)
            for i, rec in enumerate(content_recs, 1):
                print(f"{i}. {rec['title']} (相似度: {rec['similarity_score']:.3f})")
            
            # 3. 混合推荐
            print(f"\n🎯 混合推荐 - 基于《{sample_movie}》的综合推荐:")
            hybrid_recs = self.get_hybrid_recommendations(sample_movie, n=5)
            for i, rec in enumerate(hybrid_recs, 1):
                print(f"{i}. {rec['title']} (混合分数: {rec['hybrid_score']:.3f})")
        
        # 4. 系统统计信息
        print("\n📊 系统统计信息:")
        stats = self.get_system_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    
    def visualize_data(self):
        """数据可视化"""
        if self.movies_df is None:
            raise ValueError("请先加载数据")
        
        print("\n📈 生成数据可视化图表...")
        
        # 评分分布
        self.visualizer.plot_rating_distribution(self.movies_df)
        
        # 类型分布
        self.visualizer.plot_genre_distribution(self.movies_df)
        
        # 如果有推荐结果，显示推荐分数分布
        if self.is_fitted:
            try:
                sample_recs = self.get_hybrid_recommendations(n=10)
                if sample_recs:
                    self.visualizer.plot_recommendation_scores(sample_recs)
            except:
                pass
    
    def evaluate_recommendations(self, recommendations, title="推荐评估"):
        """评估推荐结果"""
        if not recommendations:
            print("没有推荐结果可评估")
            return
        
        report = self.evaluator.generate_evaluation_report(
            recommendations, len(self.movies_df) if self.movies_df is not None else None
        )
        
        print(f"\n📋 {title}:")
        print(f"推荐数量: {report['total_recommendations']}")
        print(f"多样性: {report['diversity']:.3f}")
        print(f"新颖性: {report['novelty']:.3f}")
        
        if 'coverage' in report:
            print(f"覆盖率: {report['coverage']:.3f}")
        
        if 'score_statistics' in report:
            stats = report['score_statistics']
            print(f"分数统计: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}")
    
    def get_system_statistics(self):
        """获取系统统计信息"""
        stats = {
            "数据统计": self.data_loader.get_data_summary() if self.movies_df is not None else {},
            "推荐器统计": self.hybrid_recommender.get_statistics() if self.is_fitted else {},
        }
        
        if self.deep_learning_recommender:
            stats["深度学习模型"] = self.deep_learning_recommender.get_model_summary()
        
        return stats
    
    def interactive_mode(self):
        """交互式模式"""
        if not self.is_fitted:
            print("❌ 系统尚未初始化，请先运行完整演示")
            return
        
        print("\n🎮 进入交互式推荐模式")
        print("输入 'help' 查看帮助，输入 'quit' 退出")
        
        while True:
            try:
                command = input("\n请输入命令: ").strip().lower()
                
                if command == 'quit':
                    print("👋 再见！")
                    break
                elif command == 'help':
                    self._show_help()
                elif command.startswith('content '):
                    movie_title = command[8:].strip()
                    try:
                        recs = self.get_content_recommendations(movie_title, 5)
                        print_recommendations(recs, f"基于《{movie_title}》的内容推荐")
                    except ValueError as e:
                        print(f"❌ {e}")
                elif command.startswith('hybrid '):
                    movie_title = command[7:].strip()
                    try:
                        recs = self.get_hybrid_recommendations(movie_title, n=5)
                        print_recommendations(recs, f"基于《{movie_title}》的混合推荐")
                    except Exception as e:
                        print(f"❌ 推荐生成失败: {e}")
                elif command == 'top':
                    recs = self.get_demographic_recommendations(10)
                    print("\n🏆 最受好评的电影:")
                    for i, (_, movie) in enumerate(recs.iterrows(), 1):
                        print(f"{i:2d}. {movie['title']:<30} 评分: {movie['vote_average']:.1f}")
                elif command == 'stats':
                    stats = self.get_system_statistics()
                    print("\n📊 系统统计:")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                else:
                    print("❌ 未知命令，输入 'help' 查看帮助")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 出现错误: {e}")
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
🎬 电影推荐系统命令帮助:
        
📝 基础命令:
  help              - 显示此帮助信息
  quit              - 退出程序
  top               - 显示最受好评的电影
  stats             - 显示系统统计信息
  
🎯 推荐命令:
  content <电影名>  - 基于内容的推荐
  hybrid <电影名>   - 混合推荐
  
💡 示例:
  content The Dark Knight Rises
  hybrid Inception
  
📌 注意: 电影名需要完全匹配数据库中的标题
        """
        print(help_text)

def main():
    """主函数"""
    print("🎬 电影推荐系统启动中...")
    print("="*60)
    
    try:
        # 初始化系统
        system = MovieRecommendationSystem()
        
        # 加载数据
        if not system.load_and_prepare_data():
            print("❌ 数据加载失败，程序退出")
            return
        
        # 训练推荐器
        print("\n🤖 训练推荐模型中，请稍候...")
        system.fit_all_recommenders()
        
        # 运行演示
        system.demonstrate_system()
        
        # 可视化（可选）
        try:
            system.visualize_data()
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
        
        # 交互式模式
        print("\n" + "="*60)
        choice = input("是否进入交互式模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            system.interactive_mode()
        
        print("\n✅ 程序执行完成！")
        
    except KeyboardInterrupt:
        print("\n❌ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
