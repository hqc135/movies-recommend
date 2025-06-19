"""
Web应用配置文件
"""

import os

class WebConfig:
    """Web应用配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'movie_recommender_secret_key_2025'
    
    # 调试模式
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # 主机和端口
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # 静态文件配置
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1年
    
    # 上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 缓存配置
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300
    
    # JSON配置
    JSON_AS_ASCII = False
    JSONIFY_PRETTYPRINT_REGULAR = True

class DevelopmentConfig(WebConfig):
    """开发环境配置"""
    DEBUG = True
    TESTING = False

class ProductionConfig(WebConfig):
    """生产环境配置"""
    DEBUG = False
    TESTING = False
    
    # 生产环境安全配置
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(WebConfig):
    """测试环境配置"""
    DEBUG = True
    TESTING = True

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
