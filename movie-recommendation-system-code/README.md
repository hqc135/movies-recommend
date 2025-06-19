# 🎬 智能电影推荐系统

一个基于多种算法的现代化电影推荐系统，支持人口统计学推荐、基于内容推荐、深度学习推荐和混合推荐。

## ✨ 主要特性

- 🏆 **人口统计学推荐**: 基于IMDB加权评分的高质量电影推荐
- 🎯 **基于内容推荐**: 使用TF-IDF和余弦相似度的智能内容匹配
- 🤖 **深度学习推荐**: 神经协同过滤模型的个性化推荐
- 🔄 **混合推荐**: 融合多种算法的综合推荐系统
- 🌐 **Web界面**: 现代化的Web用户界面，支持在线推荐和搜索
- 🎮 **交互式界面**: 用户友好的命令行交互体验

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器

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
   - 确保 `data/raw/` 目录包含必要的数据文件：
     - `tmdb_5000_movies.csv`
     - `tmdb_5000_credits.csv`

### 运行

#### 🖥️ 命令行版本
```bash
python main.py
```

#### 🌐 Web版本
```bash
# 推荐方式：使用启动脚本
python run_web.py

# 或者直接运行Flask应用
python app.py
```

访问: http://localhost:5000

## 📁 项目结构

```
movie-recommendation-system-code/
├── src/                    # 核心源代码
│   ├── data/              # 数据处理模块
│   ├── recommenders/      # 推荐算法模块
│   └── utils/             # 工具模块
├── templates/             # Web界面模板
├── static/                # 静态资源
├── data/                  # 数据目录
│   └── raw/               # 原始数据文件
├── tests/                 # 测试文件
├── app.py                 # Web应用主文件
├── main.py                # 命令行程序入口
├── config.py              # 统一配置文件
├── run_web.py             # Web启动脚本
└── requirements.txt       # 依赖包管理
```

## 🌐 Web界面使用指南

### 主要功能

1. **🏠 主页**
   - **系统状态检查**: 显示推荐系统是否已初始化
   - **一键初始化**: 点击按钮初始化推荐系统（首次使用）
   - **热门电影展示**: 基于评分的高质量电影推荐
   - **算法介绍**: 展示四种推荐算法的特点

2. **🔍 搜索功能**
   - **实时搜索**: 在顶部搜索框输入电影名称
   - **模糊匹配**: 支持部分匹配和关键词搜索
   - **搜索结果**: 显示匹配的电影及其详细信息

3. **🎬 电影详情页**
   - **完整信息**: 电影标题、评分、年份、类型、导演、演员等
   - **推荐功能**: 基于内容推荐和混合推荐
   - **相似电影**: 自动显示相似度最高的电影

### 快捷键
- **`/`**: 聚焦搜索框
- **`Esc`**: 清除搜索

### 常见问题

**Q: 页面显示"系统未初始化"**
A: 首次使用需要点击"初始化系统"按钮，这个过程需要几分钟时间来加载数据和训练模型。

**Q: 搜索没有结果**
A: 请检查电影名称是否正确，尝试使用关键词搜索，确保系统已经初始化完成。

**Q: 推荐功能不工作**
A: 确保系统已经初始化完成，选择的电影在数据库中存在。

## 🎯 命令行使用指南

### 交互式命令
```bash
# 🏆 查看最受好评电影
top

# 🎬 基于内容推荐
content The Dark Knight Rises

# 🎯 混合推荐
hybrid Inception

# 📊 系统统计信息
stats

# ❓ 帮助信息
help

# 🚪 退出程序
quit
```

## 🧠 推荐算法

### 1. 🏆 人口统计学推荐
- **算法**: IMDB加权评分公式
- **特点**: 推荐高质量、广受认可的电影
- **适用**: 冷启动问题、新用户推荐

### 2. 🎯 基于内容推荐
- **文本分析**: TF-IDF向量化电影描述
- **特征融合**: 结合类型、导演、演员信息
- **相似度计算**: 余弦相似度匹配
- **适用**: 基于用户已喜欢的电影推荐相似内容

### 3. 🤖 深度学习推荐
- **模型**: 神经协同过滤 (Neural Collaborative Filtering)
- **架构**: 用户/电影嵌入 + 多层神经网络
- **适用**: 大数据量的个性化推荐

### 4. 🔄 混合推荐
- **策略**: 加权融合多个推荐器
- **优势**: 结合不同算法的优点
- **适用**: 实际生产环境的综合推荐

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
```

## ⚙️ 配置

在 `config.py` 中可自定义：

```python
# 推荐参数
DEFAULT_RECOMMENDATIONS = 10    # 默认推荐数量
MAX_RECOMMENDATIONS = 20        # 最大推荐数量

# 混合推荐权重
HYBRID_WEIGHTS = {
    'deep_learning': 0.6,       # 深度学习权重
    'content_based': 0.4        # 内容推荐权重
}

# 深度学习参数
EMBEDDING_DIM = 50              # 嵌入向量维度
NEURAL_LAYERS = [128, 64, 32]   # 神经网络结构
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

本项目采用 MIT 许可证

## 🙏 致谢

- [TMDB](https://www.themoviedb.org/) 提供的电影数据
- [TensorFlow](https://tensorflow.org/) 深度学习框架
- [Flask](https://flask.palletsprojects.com/) Web框架
- [scikit-learn](https://scikit-learn.org/) 机器学习库

---

🎬 **享受您的电影推荐之旅！**
