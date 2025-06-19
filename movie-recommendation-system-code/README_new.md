# 🎬 智能电影推荐系统

一个基于多种算法的现代化电影推荐系统，经过完全模块化重构，提供高质量的个性化推荐体验。

## ✨ 主要特性

- **🏆 人口统计学推荐**: 基于IMDB加权评分的高质量电影推荐
- **🎯 基于内容推荐**: 使用TF-IDF和余弦相似度的智能内容匹配
- **🤖 深度学习推荐**: 神经协同过滤模型的个性化推荐
- **🔄 混合推荐**: 融合多种算法的综合推荐系统
- **📊 可视化分析**: 丰富的数据可视化和推荐结果展示
- **🎮 交互式界面**: 用户友好的命令行交互体验

## 🏗️ 项目结构 (重构后)

```
movie-recommendation-system-code/
├── 📁 src/                     # 核心源代码 (模块化设计)
│   ├── 📁 data/               # 数据处理模块
│   │   └── data_loader.py     # 数据加载和预处理
│   ├── 📁 recommenders/       # 推荐算法模块
│   │   ├── demographic.py     # 人口统计学推荐
│   │   ├── content_based.py   # 基于内容的推荐
│   │   ├── deep_learning.py   # 深度学习推荐
│   │   └── hybrid.py          # 混合推荐系统
│   └── 📁 utils/              # 工具模块
│       └── utils.py           # 可视化、评估、日志工具
├── 📄 config.py               # 配置管理
├── 📄 main.py                 # 主程序入口 (全新设计)
├── 📄 requirements.txt        # 依赖包管理
└── 📄 README.md              # 项目文档
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone [your-repo-url]
cd movie-recommendation-system-code

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行系统
```bash
python main.py
```

### 3. 体验功能
程序将自动：
1. 📂 加载和预处理数据
2. 🤖 训练所有推荐模型
3. 📋 展示多种推荐结果
4. 📈 生成可视化图表
5. 🎮 提供交互式推荐体验

## 🎯 使用指南

### 交互式命令
```bash
# 🏆 查看最受好评电影
top

# 🎬 基于内容推荐
content The Dark Knight Rises

# 🎯 混合推荐 (推荐!)
hybrid Inception

# 📊 系统统计信息
stats

# ❓ 帮助信息
help

# 🚪 退出程序
quit
```

### 编程接口
```python
from main import MovieRecommendationSystem

# 初始化系统
system = MovieRecommendationSystem()

# 加载数据和训练模型
system.load_and_prepare_data()
system.fit_all_recommenders()

# 获取不同类型的推荐
demographic_recs = system.get_demographic_recommendations(10)
content_recs = system.get_content_recommendations("Inception", 10)
hybrid_recs = system.get_hybrid_recommendations("Inception", n=10)

# 可视化和评估
system.visualize_data()
system.evaluate_recommendations(hybrid_recs)
```

## 🧠 算法架构

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
- **优化**: Dropout、L2正则化、批量归一化
- **适用**: 大数据量的个性化推荐

### 4. 🔄 混合推荐
- **策略**: 加权融合多个推荐器
- **优势**: 结合不同算法的优点
- **配置**: 可调节各算法权重
- **适用**: 实际生产环境的综合推荐

## ⚙️ 配置选项

在 `config.py` 中可自定义：

```python
# 🎛️ 深度学习参数
EMBEDDING_DIM = 50              # 嵌入向量维度
NEURAL_LAYERS = [128, 64, 32]   # 神经网络结构
LEARNING_RATE = 0.001           # 学习率

# ⚖️ 混合推荐权重
HYBRID_WEIGHTS = {
    'deep_learning': 0.6,       # 深度学习权重
    'content_based': 0.4,       # 内容推荐权重
    'demographic': 0.2          # 人口统计权重
}

# 📊 推荐参数
DEFAULT_RECOMMENDATIONS = 10    # 默认推荐数量
MAX_RECOMMENDATIONS = 20        # 最大推荐数量
```

## 📊 评估指标

系统提供多维度评估：

- **🎯 准确性**: 预测评分的RMSE/MAE
- **🌈 多样性**: 推荐结果的类型多样性
- **✨ 新颖性**: 基于流行度的新颖性分数
- **📈 覆盖率**: 推荐系统的项目覆盖范围

## 🔧 技术亮点

### 代码质量
- ✅ **模块化设计**: 清晰的关注点分离
- ✅ **类型注解**: 完整的类型提示
- ✅ **文档字符串**: 详细的函数说明
- ✅ **错误处理**: 完善的异常处理机制
- ✅ **日志系统**: 结构化的日志记录

### 性能优化
- ⚡ **缓存机制**: 推荐结果缓存
- ⚡ **向量化计算**: 高效的矩阵运算
- ⚡ **批处理**: 优化的数据处理
- ⚡ **内存管理**: 智能的内存使用

### 可扩展性
- 🔌 **插件架构**: 易于添加新的推荐算法
- 🔌 **配置驱动**: 灵活的参数配置
- 🔌 **API接口**: 标准化的调用接口

## 📋 系统要求

- **Python**: 3.8+
- **内存**: 4GB+ (推荐8GB)
- **存储**: 1GB+ 可用空间
- **依赖**: 详见 requirements.txt

## 🔍 故障排除

### 常见问题

1. **TensorFlow 安装失败**
   ```bash
   # 解决方案: 使用CPU版本
   pip install tensorflow-cpu
   ```

2. **内存不足错误**
   ```python
   # 解决方案: 在config.py中减少数据量
   BATCH_SIZE = 128  # 降低批次大小
   ```

3. **路径错误**
   ```python
   # 解决方案: 检查config.py中的路径设置
   DATA_PATH = "你的数据路径"
   ```

## 🚀 未来规划

- [ ] 🌐 Web界面开发
- [ ] 📱 RESTful API接口
- [ ] 🔄 实时推荐系统
- [ ] 🎯 A/B测试框架
- [ ] 📈 更多评估指标
- [ ] 🤖 强化学习算法

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/AmazingFeature`
3. 提交更改: `git commit -m 'Add some AmazingFeature'`
4. 推送分支: `git push origin feature/AmazingFeature`
5. 提交Pull Request

## 📝 版本历史

### v2.0.0 - 模块化重构版 🎉
- ✅ 完全重构为模块化架构
- ✅ 新增交互式命令行界面
- ✅ 增强的混合推荐系统
- ✅ 改进的深度学习模型
- ✅ 完善的可视化和评估工具

### v1.0.0 - 初始版本
- ✅ 基础推荐算法实现
- ✅ 数据加载和处理
- ✅ 基本可视化功能

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📞 支持

- 🐛 **Bug报告**: 在GitHub Issues中提交
- 💡 **功能建议**: 在GitHub Discussions中讨论  
- 📧 **技术支持**: [your-email@example.com]

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！
