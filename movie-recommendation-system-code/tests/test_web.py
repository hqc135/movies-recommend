"""
Web应用测试
"""
import pytest
import json

def test_index_page(client):
    """测试主页"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'movie' in response.data.lower()

def test_health_check(client):
    """测试健康检查端点"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_api_movies(client):
    """测试电影API"""
    response = client.get('/api/movies')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'movies' in data

def test_recommendation_api(client, sample_data):
    """测试推荐API"""
    response = client.post('/api/recommend', 
                          json={'method': 'demographic', 'count': 5})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'recommendations' in data
