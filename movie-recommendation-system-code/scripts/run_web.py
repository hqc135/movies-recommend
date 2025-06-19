#!/usr/bin/env python3
"""
电影推荐系统Web界面启动脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """检查数据文件是否存在"""
    try:
        from config import Config
        
        config = Config()
        data_files = [
            (config.MOVIES_FILE, "电影数据文件"),
            (config.CREDITS_FILE, "演职人员数据文件")
        ]
        
        missing_files = []
        
        for file_path, description in data_files:
            if not os.path.exists(file_path):
                missing_files.append((file_path, description))
        
        if missing_files:
            print("❌ 缺少以下数据文件:")
            for file_path, description in missing_files:
                print(f"   - {description}: {file_path}")
            print("\n请确保数据文件存在于正确的路径中")
            return False
        
        return True
    except ImportError as e:
        print(f"❌ 导入配置文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🎬 电影推荐系统Web界面")
    print("=" * 50)
    
    # 检查依赖
    print("📦 检查依赖包...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 依赖包检查通过")
    
    # 检查数据文件
    print("📁 检查数据文件...")
    if not check_data_files():
        sys.exit(1)
    print("✅ 数据文件检查通过")
    
    # 导入并启动应用
    print("🚀 启动Web应用...")
    print("📍 访问地址: http://localhost:5000")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n✅ Web服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()