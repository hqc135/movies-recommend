"""
测试配置文件
"""
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def app():
    """Flask应用测试夹具"""
    from app import create_app
    app = create_app(testing=True)
    app.config.update({
        "TESTING": True,
    })
    return app

@pytest.fixture
def client(app):
    """测试客户端"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """测试命令行运行器"""
    return app.test_cli_runner()

@pytest.fixture
def sample_data():
    """示例数据"""
    return {
        'movies': [
            {'id': 1, 'title': 'Test Movie 1', 'genres': 'Action|Adventure'},
            {'id': 2, 'title': 'Test Movie 2', 'genres': 'Comedy|Romance'},
        ],
        'ratings': [
            {'userId': 1, 'movieId': 1, 'rating': 4.5},
            {'userId': 1, 'movieId': 2, 'rating': 3.0},
        ]
    }
