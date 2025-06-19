# 项目结构说明

## 📁 重构后的目录结构

```
movie-recommendation-system-code/
├── 📁 src/                          # 核心源代码
│   ├── 📁 data/                    # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py          # 数据加载和预处理
│   │   └── user_feedback.py        # 用户反馈处理
│   ├── 📁 recommenders/            # 推荐算法模块
│   │   ├── __init__.py
│   │   ├── demographic.py          # 人口统计学推荐
│   │   ├── content_based.py        # 基于内容的推荐
│   │   ├── deep_learning.py        # 深度学习推荐
│   │   └── hybrid.py               # 混合推荐系统
│   └── 📁 utils/                   # 工具模块
│       ├── __init__.py
│       └── utils.py                # 可视化、评估、日志工具
├── 📁 config/                      # 配置文件
│   ├── __init__.py
│   ├── config.py                   # 主配置文件
│   └── web_config.py               # Web配置
├── 📁 data/                        # 数据目录
│   ├── 📁 raw/                     # 原始数据
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── 📁 processed/               # 处理后的数据
├── 📁 models/                      # 模型文件
│   └── (训练好的模型文件)
├── 📁 notebooks/                   # Jupyter笔记本
│   └── Movie Recommendation.ipynb
├── 📁 scripts/                     # 脚本文件
│   ├── run_web.py                  # Web启动脚本
│   └── demo_web.py                 # 演示脚本
├── 📁 tests/                       # 测试文件
│   ├── conftest.py                 # 测试配置
│   ├── test_web.py                 # Web测试
│   └── test_recommenders.py        # 推荐器测试
├── 📁 docs/                        # 文档
│   ├── README_new.md               # 详细文档
│   ├── WEB_GUIDE.md               # Web使用指南
│   └── PROJECT_STRUCTURE.md        # 本文件
├── 📁 templates/                   # Web模板
│   ├── base.html
│   ├── index.html
│   ├── movie_detail.html
│   └── 404.html
├── 📁 static/                      # 静态资源
│   ├── 📁 css/
│   │   └── style.css
│   └── 📁 js/
│       ├── main.js
│       └── index.js
├── 📁 logs/                        # 日志文件
├── 📄 app.py                       # Web应用主文件
├── 📄 main.py                      # 命令行程序入口
├── 📄 requirements.txt             # 依赖包
├── 📄 setup.py                     # 安装配置
├── 📄 .gitignore                   # Git忽略文件
├── 📄 Dockerfile                   # Docker配置
├── 📄 docker-compose.yml           # Docker Compose配置
└── 📄 README.md                    # 项目说明
```

## 🔄 主要变更

### 1. 数据管理优化
- **原始数据**: `data/raw/` - 存放未处理的数据文件
- **处理数据**: `data/processed/` - 存放预处理后的数据
- **模型文件**: `models/` - 统一存放训练好的模型

### 2. 配置管理
- **集中配置**: `config/` 目录统一管理所有配置文件
- **路径更新**: 配置文件中的路径已更新为相对路径

### 3. 代码组织
- **测试分离**: `tests/` 目录包含所有测试文件
- **脚本整理**: `scripts/` 目录存放启动和演示脚本
- **文档集中**: `docs/` 目录统一管理文档

### 4. 部署支持
- **Docker**: 添加 Dockerfile 和 docker-compose.yml
- **包管理**: 添加 setup.py 支持pip安装
- **环境隔离**: .gitignore 文件优化

## 🚀 使用方式

### 开发模式
```bash
# 安装依赖
pip install -r requirements.txt

# 运行命令行版本
python main.py

# 运行Web版本
python app.py
```

### 生产部署
```bash
# Docker部署
docker-compose up -d

# 或者包安装
pip install -e .
movie-recommend-web
```

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_web.py -v
```

## 📋 迁移注意事项

1. **导入路径**: 所有模块的导入路径已更新
2. **配置文件**: config.py 移动到 config/ 目录
3. **数据路径**: 数据文件路径配置已更新为相对路径
4. **日志文件**: 日志将统一保存到 logs/ 目录

## 🛠️ 开发建议

1. **代码规范**: 使用 black、flake8 进行代码格式化
2. **测试驱动**: 为新功能编写对应的测试
3. **文档更新**: 及时更新相关文档
4. **版本控制**: 使用语义化版本号管理
