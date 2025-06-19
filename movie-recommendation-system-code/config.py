"""
配置文件 - 管理项目的所有配置参数
"""
import os

class Config:
    """项目配置类"""
    
    # ======================== 数据路径配置 ========================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = "C:/Users/18304/OneDrive - whu.edu.cn/桌面/movies recommend/archive"
    MOVIES_FILE = os.path.join(DATA_PATH, "tmdb_5000_movies.csv")
    CREDITS_FILE = os.path.join(DATA_PATH, "tmdb_5000_credits.csv")
    
    # ======================== 模型参数配置 ========================
    # 深度学习模型参数
    EMBEDDING_DIM = 50
    NEURAL_LAYERS = [128, 64, 32]
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    
    # 训练参数
    EPOCHS = 20
    BATCH_SIZE = 256
    VALIDATION_SPLIT = 0.2
    
    # ======================== 推荐参数配置 ========================
    DEFAULT_RECOMMENDATIONS = 10
    MAX_RECOMMENDATIONS = 20
    
    # 混合推荐权重
    HYBRID_WEIGHTS = {
        'deep_learning': 0.6,
        'content_based': 0.4
    }
    
    # ======================== 数据处理参数 ========================
    # IMDB加权评分参数
    VOTE_COUNT_QUANTILE = 0.9  # 投票数阈值分位数
    
    # 特征提取参数
    MAX_FEATURES_PER_CATEGORY = 3  # 每个类别最多提取的特征数
    
    # ======================== 可视化配置 ========================
    FIGURE_SIZE = (12, 6)
    PLOT_STYLE = 'seaborn'
    
    # ======================== 缓存配置 ========================
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    ENABLE_CACHE = True
    
    # ======================== 日志配置 ========================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def validate_paths(cls):
        """验证路径是否存在"""
        paths_to_check = [cls.DATA_PATH, cls.MOVIES_FILE, cls.CREDITS_FILE]
        for path in paths_to_check:
            if not os.path.exists(path):
                raise FileNotFoundError(f"路径不存在: {path}")
        return True
