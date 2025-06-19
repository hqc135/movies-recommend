# 🎬 智能电影推荐系统

一个基于多种算法的现代化电影推荐系统，支持人口统计学推荐、基于内容推荐、深度学习推荐和混合推荐。

## ✨ 主要特性

- 🏆 **人口统计学推荐**: 基于IMDB加权评分的高质量电影推荐
- 🎯 **基于内容推荐**: 使用TF-IDF和余弦相似度的智能内容匹配
- 🤖 **深度学习推荐**: 神经协同过滤模型的个性化推荐
- 🔄 **混合推荐**: 融合多种算法的综合推荐系统
- 📊 **可视化分析**: 丰富的数据可视化和推荐结果展示
- 🎮 **交互式界面**: 用户友好的命令行交互体验
- 🌐 **Web界面**: 现代化的Web用户界面，支持在线推荐和搜索

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器
- (可选) CUDA 11.2+ 用于GPU加速

### 安装

1. **克隆项目**
```bash
git clone <repository-url>
cd movie-recommendation-system-code
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据**
   - 确保 `data/raw/` 目录包含必要的数据文件
   - 运行数据预处理（如果需要）

### 运行

#### 命令行版本
```bash
python main.py
```

#### Web版本
```bash
python app.py
```
然后访问 http://localhost:5000

#### Docker部署
```bash
docker-compose up -d
```

## 📁 项目结构

```
├── src/                    # 核心源代码
├── config/                 # 配置文件
├── data/                   # 数据目录
├── models/                 # 模型文件
├── tests/                  # 测试文件
├── docs/                   # 文档
├── templates/              # Web模板
├── static/                 # 静态资源
├── logs/                   # 日志文件
└── scripts/                # 脚本文件
```

详细结构说明请参考 [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## 🎯 推荐算法

### 1. 人口统计学推荐
基于电影的整体受欢迎程度和评分，推荐高质量的热门电影。

### 2. 基于内容推荐
- **TF-IDF**: 基于电影描述的文本相似度
- **元数据**: 基于导演、演员、类型等元数据的相似度

### 3. 深度学习推荐
使用神经协同过滤模型，学习用户和电影的潜在特征进行推荐。

### 4. 混合推荐
结合多种算法的优势，提供更准确和多样化的推荐结果。

## 🛠️ 开发

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_web.py -v

# 生成覆盖率报告
pytest --cov=src tests/
```

### 代码质量
```bash
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 类型检查
mypy src/
```

## 📊 性能指标

- **准确率**: 推荐算法的预测准确性
- **多样性**: 推荐结果的多样化程度
- **响应时间**: 推荐请求的处理时间
- **覆盖率**: 推荐系统对电影库的覆盖程度

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 文档: [docs/](docs/)
- 问题反馈: [GitHub Issues]

## 🙏 致谢

- [TMDB](https://www.themoviedb.org/) 提供的电影数据
- [TensorFlow](https://tensorflow.org/) 深度学习框架
- [Flask](https://flask.palletsprojects.com/) Web框架
- [scikit-learn](https://scikit-learn.org/) 机器学习库
