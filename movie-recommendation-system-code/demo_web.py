#!/usr/bin/env python3
"""
电影推荐系统演示脚本
展示Web界面的主要功能
"""

import time
import requests
import json
from urllib.parse import quote

def demo_web_interface():
    """演示Web界面功能"""
    base_url = "http://localhost:5000"
    
    print("🎬 电影推荐系统Web界面演示")
    print("=" * 50)
    
    # 1. 检查系统状态
    print("1️⃣ 检查系统状态...")
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ 系统状态: {'已初始化' if status['initialized'] else '未初始化'}")
            print(f"   📊 电影数量: {status.get('total_movies', 0)}")
            print(f"   🤖 深度学习: {'可用' if status['deep_learning_available'] else '不可用'}")
        else:
            print(f"   ❌ 无法连接到服务器 (状态码: {response.status_code})")
            return
    except requests.exceptions.RequestException as e:
        print(f"   ❌ 连接失败: {e}")
        print(f"   💡 请确保Web服务器正在运行: python run_web.py")
        return
    
    # 2. 如果系统未初始化，进行初始化
    if not status['initialized']:
        print("\n2️⃣ 初始化推荐系统...")
        print("   ⏳ 正在初始化，请耐心等待...")
        try:
            response = requests.get(f"{base_url}/initialize", timeout=300)  # 5分钟超时
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    print("   ✅ 系统初始化成功!")
                else:
                    print(f"   ❌ 初始化失败: {result.get('message', '未知错误')}")
                    return
            else:
                print(f"   ❌ 初始化请求失败 (状态码: {response.status_code})")
                return
        except requests.exceptions.RequestException as e:
            print(f"   ❌ 初始化失败: {e}")
            return
    
    # 3. 获取热门电影
    print("\n3️⃣ 获取热门电影...")
    try:
        response = requests.get(f"{base_url}/top-movies", timeout=30)
        if response.status_code == 200:
            movies = response.json()
            print(f"   📽️ 获取到 {len(movies)} 部热门电影")
            print("   🏆 前5部电影:")
            for i, movie in enumerate(movies[:5], 1):
                print(f"      {i}. {movie['title']} (评分: {movie['rating']}, 年份: {movie['year']})")
        else:
            print(f"   ❌ 获取热门电影失败 (状态码: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求失败: {e}")
    
    # 4. 搜索电影
    print("\n4️⃣ 搜索电影演示...")
    search_queries = ["Batman", "Inception", "Avatar"]
    
    for query in search_queries:
        print(f"   🔍 搜索: '{query}'")
        try:
            response = requests.get(f"{base_url}/search", params={'q': query}, timeout=30)
            if response.status_code == 200:
                results = response.json()
                if results:
                    print(f"      ✅ 找到 {len(results)} 个结果")
                    for result in results[:3]:  # 只显示前3个
                        print(f"         - {result['title']} ({result['year']}) - 评分: {result['rating']}")
                else:
                    print(f"      ℹ️ 没有找到相关电影")
            else:
                print(f"      ❌ 搜索失败 (状态码: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"      ❌ 搜索请求失败: {e}")
        
        time.sleep(1)  # 避免请求过快
    
    # 5. 获取推荐
    print("\n5️⃣ 获取推荐演示...")
    test_movie = "Inception"
    recommendation_types = [
        ("content", "基于内容推荐"),
        ("hybrid", "混合推荐")
    ]
    
    for rec_type, rec_name in recommendation_types:
        print(f"   🎯 {rec_name} - 基于电影: '{test_movie}'")
        try:
            params = {'movie': test_movie, 'type': rec_type}
            response = requests.get(f"{base_url}/recommend", params=params, timeout=30)
            if response.status_code == 200:
                recommendations = response.json()
                if isinstance(recommendations, list) and recommendations:
                    print(f"      ✅ 获取到 {len(recommendations)} 个推荐")
                    for i, rec in enumerate(recommendations[:3], 1):  # 只显示前3个
                        print(f"         {i}. {rec['title']} (推荐分数: {rec['score']}, 评分: {rec['rating']})")
                else:
                    print(f"      ℹ️ 没有推荐结果")
            else:
                print(f"      ❌ 获取推荐失败 (状态码: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"      ❌ 推荐请求失败: {e}")
        
        time.sleep(1)
    
    # 6. 总结
    print("\n🎉 演示完成!")
    print("=" * 50)
    print("📱 现在您可以:")
    print("   • 在浏览器中访问: http://localhost:5000")
    print("   • 搜索您喜欢的电影")
    print("   • 获取个性化推荐")
    print("   • 浏览电影详情页面")
    print("   • 体验现代化的Web界面")

def main():
    """主函数"""
    try:
        demo_web_interface()
    except KeyboardInterrupt:
        print("\n\n❌ 演示被中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()
