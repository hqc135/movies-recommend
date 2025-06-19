"""
项目配置文件 - 统一管理所有配置参数
"""
import os

class Config:
    """项目配置类"""
    
    # ======================== 基础路径配置 ========================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
    MOVIES_FILE = os.path.join(DATA_PATH, "tmdb_5000_movies.csv")
    CREDITS_FILE = os.path.join(DATA_PATH, "tmdb_5000_credits.csv")
    
    # ======================== Web应用配置 ========================
    DEBUG = True
    SECRET_KEY = 'dev-secret-key-change-in-production'
    
    # ======================== 推荐系统参数 ========================
    DEFAULT_RECOMMENDATIONS = 10
    MAX_RECOMMENDATIONS = 20
    VOTE_COUNT_QUANTILE = 0.9
    MAX_FEATURES_PER_CATEGORY = 3
    
    # 混合推荐权重
    HYBRID_WEIGHTS = {
        'deep_learning': 0.6,
        'content_based': 0.4
    }
    
    # ======================== 深度学习模型参数 ========================
    EMBEDDING_DIM = 50
    NEURAL_LAYERS = [128, 64, 32]
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    EPOCHS = 20
    BATCH_SIZE = 256
    VALIDATION_SPLIT = 0.2
    
    # ======================== 路径验证 ========================
    @classmethod
    def validate_paths(cls):
        """验证关键路径是否存在"""
        paths_to_check = [cls.DATA_PATH, cls.MOVIES_FILE, cls.CREDITS_FILE]
        for path in paths_to_check:
            if not os.path.exists(path):
                raise FileNotFoundError(f"路径不存在: {path}")
        return True
