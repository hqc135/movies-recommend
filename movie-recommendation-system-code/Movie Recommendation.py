"""
电影推荐系统 - 多算法集成版本
使用人口统计学过滤、基于内容过滤和深度学习推荐
"""

# 基础库导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# 深度学习库导入
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# ======================== 数据加载 ========================
path = "C:/Users/18304/OneDrive - whu.edu.cn/桌面/movies recommend/archive"
credits_df = pd.read_csv(path + "/tmdb_5000_credits.csv")
movies_df = pd.read_csv(path + "/tmdb_5000_movies.csv")
credits_df.columns = ['id','tittle','cast','crew']
movies_df = movies_df.merge(credits_df, on="id")

# ======================== 人口统计学推荐 ========================
# IMDB加权评分：WR = (v/(v+m)) * R + (m/(v+m)) * C
C = movies_df["vote_average"].mean()  # 所有电影的平均评分
m = movies_df["vote_count"].quantile(0.9)  # 90%分位数的投票数阈值
new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]


def weighted_rating(x, C=C, m=m):
    """计算IMDB加权评分"""
    v = x["vote_count"]  # 投票数
    R = x["vote_average"]  # 平均评分
    return (v/(v + m) * R) + (m/(v + m) * C)

new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=False)
top_movies = new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)


# 流行度可视化函数
def plot():
    """绘制最受欢迎的10部电影"""
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 movies")
    plt.xlabel("Popularity")
    plt.show()

plot() 

# ======================== 基于内容的推荐 ========================
movies_df["overview"] = movies_df["overview"].fillna("")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

# 计算余弦相似度矩阵
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    """
    基于余弦相似度的电影推荐函数
    1. 获取目标电影的索引
    2. 计算与所有电影的相似度分数
    3. 按相似度排序，取前10部电影
    4. 返回推荐电影标题列表
    """
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 排除自身，取前10个
    
    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return movies




# 基于内容推荐结果展示
print("基于内容的推荐结果:")
print("\n《The Dark Knight Rises》相似电影:")
print(get_recommendations("The Dark Knight Rises"))
print("\n《The Avengers》相似电影:")
print(get_recommendations("The Avengers"))

# ======================== 基于元数据的内容推荐 ========================
features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

def get_director(x):
    """从crew数据中提取导演信息"""
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    """提取前3个名称（演员、关键词等）"""
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

# 提取特征
movies_df["director"] = movies_df["crew"].apply(get_director)
features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

def clean_data(x):
    """清理文本数据：转小写、去空格"""
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


# 应用数据清理
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)

def create_soup(x):
    """将所有特征组合成一个字符串"""
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies_df["soup"] = movies_df.apply(create_soup, axis=1)

# 基于元数据的向量化和相似度计算
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])

# 元数据推荐结果展示
print("\n################ 基于元数据的内容推荐 #############")
print("《The Dark Knight Rises》推荐:")
print(get_recommendations("The Dark Knight Rises", cosine_sim2))
print("\n《The Avengers》推荐:")
print(get_recommendations("The Avengers", cosine_sim2))


# ======================== 深度学习推荐系统 ========================

class DeepLearningRecommendationSystem:
    """基于神经网络的电影推荐系统 - 神经协同过滤(NCF)"""
    
    def __init__(self, embedding_size=50, hidden_units=[128, 64, 32]):
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.model = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
    def prepare_data(self, movies_df):
        """准备训练数据 - 基于电影评分和流行度创建虚拟用户-电影交互数据"""
        print("正在准备深度学习训练数据...")
        np.random.seed(42)
        user_movie_interactions = []
        
        for idx, movie in movies_df.iterrows():
            # 基于电影评分和投票数生成虚拟用户数
            popularity_factor = min(movie['vote_count'] / 100, 50)
            num_users = max(int(popularity_factor), 5)
            
            for user_id in range(num_users):
                # 基于真实评分添加噪声生成虚拟评分
                base_rating = movie['vote_average']
                noise = np.random.normal(0, 0.5)
                rating = max(1, min(10, base_rating + noise))
                
                user_movie_interactions.append({
                    'user_id': f'user_{idx}_{user_id}',
                    'movie_id': movie['id'],
                    'rating': rating,
                    'movie_title': movie['title']
                })
        
        interactions_df = pd.DataFrame(user_movie_interactions)
        interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['movie_encoded'] = self.movie_encoder.fit_transform(interactions_df['movie_id'])
        
        print(f"生成了 {len(interactions_df)} 个用户-电影交互记录")
        print(f"用户数: {interactions_df['user_encoded'].nunique()}")
        print(f"电影数: {interactions_df['movie_encoded'].nunique()}")
        return interactions_df
    
    def build_model(self, num_users, num_movies):
        """构建神经协同过滤模型"""
        print("构建神经协同过滤模型...")
        
        # 用户嵌入层
        user_input = keras.layers.Input(shape=(), name='user_id')
        user_embedding = keras.layers.Embedding(num_users, self.embedding_size, name='user_embedding')(user_input)
        user_vec = keras.layers.Flatten(name='user_flatten')(user_embedding)
        
        # 电影嵌入层
        movie_input = keras.layers.Input(shape=(), name='movie_id')
        movie_embedding = keras.layers.Embedding(num_movies, self.embedding_size, name='movie_embedding')(movie_input)
        movie_vec = keras.layers.Flatten(name='movie_flatten')(movie_embedding)
        
        # 特征融合和多层神经网络
        concat = keras.layers.Concatenate(name='concat')([user_vec, movie_vec])
        dense = concat
        for units in self.hidden_units:
            dense = keras.layers.Dense(units, activation='relu')(dense)
            dense = keras.layers.Dropout(0.2)(dense)
        # 输出层 - 评分预测
        output = keras.layers.Dense(1, activation='linear', name='rating_output')(dense)
        
        # 创建和编译模型
        model = keras.Model(inputs=[user_input, movie_input], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, interactions_df, epochs=50, batch_size=256, validation_split=0.2):
        """训练模型"""
        print("开始训练深度学习模型...")
        
        # 准备训练数据
        X_users = interactions_df['user_encoded'].values
        X_movies = interactions_df['movie_encoded'].values
        y_ratings = interactions_df['rating'].values
        
        # 分割训练和验证数据
        X_users_train, X_users_val, X_movies_train, X_movies_val, y_train, y_val = train_test_split(
            X_users, X_movies, y_ratings, test_size=validation_split, random_state=42
        )
        
        # 训练模型
        history = self.model.fit(
            [X_users_train, X_movies_train], y_train,
            validation_data=([X_users_val, X_movies_val], y_val),
            epochs=epochs, batch_size=batch_size, verbose=1
        )
        
        print("模型训练完成!")
        return history
    
    def predict_rating(self, user_id, movie_id):
        """预测用户对电影的评分"""
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
            movie_encoded = self.movie_encoder.transform([movie_id])[0]
            rating = self.model.predict([np.array([user_encoded]), np.array([movie_encoded])])[0][0]
            return max(1, min(10, rating))
        except:
            return 5.0
    
    def recommend_movies(self, user_id, movies_df, top_n=10):
        """为用户推荐电影"""
        movie_ratings = []
        
        for idx, movie in movies_df.iterrows():
            predicted_rating = self.predict_rating(user_id, movie['id'])
            movie_ratings.append({
                'title': movie['title'],
                'movie_id': movie['id'],
                'predicted_rating': predicted_rating,
                'actual_rating': movie['vote_average'],
                'genres': movie.get('genres', 'N/A')
            })
        
        # 按预测评分排序并返回top-n
        movie_ratings.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return movie_ratings[:top_n]
    
    def get_movie_embeddings(self):
        """获取电影的嵌入向量，可用于相似度计算"""
        if self.model is None:
            return None
        movie_embedding_layer = self.model.get_layer('movie_embedding')
        return movie_embedding_layer.get_weights()[0]

# ======================== 深度学习演示函数 ========================

def demo_deep_learning_recommendation():
    """深度学习推荐系统演示"""
    print("\n" + "="*60)
    print("           深度学习电影推荐系统演示")
    print("="*60)
    
    # 初始化并训练模型
    dl_recommender = DeepLearningRecommendationSystem()
    interactions_df = dl_recommender.prepare_data(movies_df.head(100))
    
    num_users = interactions_df['user_encoded'].nunique()
    num_movies = interactions_df['movie_encoded'].nunique()
    model = dl_recommender.build_model(num_users, num_movies)
    
    print("\n模型架构:")
    model.summary()
    
    # 训练模型
    history = dl_recommender.train(interactions_df, epochs=20, batch_size=128)
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 生成推荐并展示结果
    print("\n深度学习推荐结果:")
    sample_user = interactions_df['user_id'].iloc[0]
    recommendations = dl_recommender.recommend_movies(sample_user, movies_df.head(100), top_n=10)
    
    print(f"\n为用户 {sample_user} 推荐的电影:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec['title']:<30} 预测评分: {rec['predicted_rating']:.2f} 实际评分: {rec['actual_rating']:.1f}")
    
    return dl_recommender

# ======================== 混合推荐系统 ========================

class HybridRecommendationSystem:
    """混合推荐系统 - 结合基于内容和深度学习的推荐"""
    
    def __init__(self, dl_recommender, content_sim_matrix, weight_dl=0.6, weight_content=0.4):
        self.dl_recommender = dl_recommender
        self.content_sim_matrix = content_sim_matrix
        self.weight_dl = weight_dl
        self.weight_content = weight_content
    
    def hybrid_recommend(self, user_id, movie_title, movies_df, top_n=10):
        """混合推荐：结合深度学习和基于内容的推荐"""
        # 深度学习推荐
        dl_recommendations = self.dl_recommender.recommend_movies(user_id, movies_df, top_n=20)
        dl_scores = {rec['title']: rec['predicted_rating'] for rec in dl_recommendations}
        
        # 基于内容的推荐
        content_recommendations = get_recommendations(movie_title, self.content_sim_matrix)
        
        # 合并分数
        hybrid_scores = {}
        
        # 添加深度学习分数 (归一化到0-1)
        for title, score in dl_scores.items():
            hybrid_scores[title] = self.weight_dl * (score / 10.0)
        
        # 添加基于内容的分数 (基于排名)
        for i, title in enumerate(content_recommendations):
            content_score = (20 - i) / 20.0
            if title in hybrid_scores:
                hybrid_scores[title] += self.weight_content * content_score
            else:
                hybrid_scores[title] = self.weight_content * content_score
        
        # 排序并返回top-n
        sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [{'title': title, 'hybrid_score': score} for title, score in sorted_recommendations[:top_n]]

# ======================== 主程序执行 ========================

if __name__ == "__main__":
    """运行推荐系统演示"""
    try:
        # 运行深度学习推荐系统
        dl_system = demo_deep_learning_recommendation()
        print("\n✅ 深度学习推荐系统演示完成!")
        
        # 混合推荐系统演示
        hybrid_system = HybridRecommendationSystem(dl_system, cosine_sim2)
        sample_user = "user_0_0"
        hybrid_recs = hybrid_system.hybrid_recommend(
            sample_user, "The Dark Knight Rises", movies_df.head(100), top_n=10
        )
        
        print(f"\n🎯 混合推荐系统为用户推荐的电影:")
        for i, rec in enumerate(hybrid_recs, 1):
            print(f"{i:2d}. {rec['title']:<30} 混合分数: {rec['hybrid_score']:.3f}")
            
    except ImportError:
        print("\n⚠️  需要安装 TensorFlow 才能运行深度学习模型")
        print("请运行: pip install tensorflow")
    except Exception as e:
        print(f"\n❌ 运行深度学习模型时出错: {e}")
        print("请检查数据和环境配置")






