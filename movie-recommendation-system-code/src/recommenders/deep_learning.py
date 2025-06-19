"""
深度学习推荐模块
实现神经协同过滤(Neural Collaborative Filtering)
"""
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    
    # 配置GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"找到 {len(physical_devices)} 个GPU设备")
        # 启用GPU内存增长，避免占用所有GPU内存
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU配置完成，启用内存增长模式")
    else:
        print("未检测到GPU设备，将使用CPU进行计算")
        
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow未安装，深度学习功能将不可用")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from config import Config

logger = logging.getLogger(__name__)

class DeepLearningRecommender:
    """基于神经网络的电影推荐系统"""
    
    def __init__(self, embedding_size: int = 50, hidden_units: Optional[List[int]] = None, 
                 learning_rate: float = 0.001, dropout_rate: float = 0.2, use_mixed_precision: bool = True):
        """
        初始化深度学习推荐器
        
        Args:
            embedding_size: 嵌入向量维度
            hidden_units: 隐藏层单元数列表
            learning_rate: 学习率
            dropout_rate: Dropout比率
            use_mixed_precision: 是否使用混合精度训练(GPU加速)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("需要安装TensorFlow才能使用深度学习推荐器")
        
        # 配置混合精度训练
        if use_mixed_precision and len(tf.config.list_physical_devices('GPU')) > 0:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("启用混合精度训练以提高GPU性能")
            
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units if hidden_units is not None else [128, 64, 32]
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_mixed_precision = use_mixed_precision
        
        self.model = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.training_history = None
        
        # 设置随机种子以保证结果可重复
        tf.random.set_seed(42)
        
        # 显示GPU状态
        self.print_gpu_status()
    
    def print_gpu_status(self):
        """显示GPU状态信息"""
        print("\n=== GPU状态信息 ===")
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(f"可用GPU数量: {len(physical_devices)}")
            for i, device in enumerate(physical_devices):
                print(f"GPU {i}: {device}")
                # 检查GPU内存信息
                try:
                    memory_info = tf.config.experimental.get_memory_info(device.name.replace('/physical_device:', '/device:'))
                    print(f"  当前内存使用: {memory_info['current'] / 1024**3:.2f} GB")
                    print(f"  峰值内存使用: {memory_info['peak'] / 1024**3:.2f} GB")
                except:
                    print(f"  无法获取内存信息")
            
            # 检查混合精度状态
            policy = tf.keras.mixed_precision.global_policy()
            print(f"混合精度策略: {policy.name}")
        else:
            print("未检测到GPU设备，使用CPU计算")
        print("==================\n")
        
    def prepare_interaction_data(self, movies_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
        """
        准备用户-电影交互数据
        
        Args:
            movies_df: 电影数据框
            min_interactions: 每部电影最少交互数
            
        Returns:
            用户-电影交互数据框
        """
        logger.info("正在生成用户-电影交互数据...")
        
        interactions = []
        
        for idx, movie in movies_df.iterrows():
            # 基于电影的受欢迎程度生成用户交互
            vote_count = movie.get('vote_count', 0)
            vote_average = movie.get('vote_average', 5.0)
            popularity = movie.get('popularity', 0)
            
            # 计算应该生成的交互数量
            interaction_count = max(
                min_interactions,
                min(int(vote_count / 10), 100)  # 限制最大交互数
            )
            
            # 生成虚拟用户交互
            for user_idx in range(interaction_count):
                # 基于真实评分生成虚拟评分（添加噪声）
                base_rating = vote_average
                noise = np.random.normal(0, 0.8)
                rating = np.clip(base_rating + noise, 1.0, 10.0)
                
                # 考虑流行度影响
                popularity_factor = min(popularity / 50, 2.0)
                rating += popularity_factor * np.random.uniform(-0.5, 0.5)
                rating = np.clip(rating, 1.0, 10.0)
                
                interactions.append({
                    'user_id': f'user_{idx}_{user_idx}',
                    'movie_id': movie['id'],
                    'movie_title': movie['title'],
                    'rating': rating,
                    'original_rating': vote_average,
                    'popularity': popularity
                })
        
        interactions_df = pd.DataFrame(interactions)
        
        # 编码用户和电影ID
        interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['movie_encoded'] = self.movie_encoder.fit_transform(interactions_df['movie_id'])
        
        logger.info(f"生成了 {len(interactions_df)} 个交互记录")
        logger.info(f"用户数: {interactions_df['user_encoded'].nunique()}")
        logger.info(f"电影数: {interactions_df['movie_encoded'].nunique()}")
        
        return interactions_df
    
    def build_model(self, num_users: int, num_movies: int) -> keras.Model:
        """
        构建神经协同过滤模型
        
        Args:
            num_users: 用户数量
            num_movies: 电影数量
            
        Returns:
            Keras模型
        """
        logger.info("构建神经协同过滤模型...")
        
        # 用户输入和嵌入
        user_input = keras.layers.Input(shape=(), name='user_id', dtype='int32')
        user_embedding = keras.layers.Embedding(
            num_users, self.embedding_size, 
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='user_embedding'
        )(user_input)
        user_vec = keras.layers.Flatten(name='user_flatten')(user_embedding)
        
        # 电影输入和嵌入
        movie_input = keras.layers.Input(shape=(), name='movie_id', dtype='int32')
        movie_embedding = keras.layers.Embedding(
            num_movies, self.embedding_size,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='movie_embedding'
        )(movie_input)
        movie_vec = keras.layers.Flatten(name='movie_flatten')(movie_embedding)
        
        # 特征融合
        concat = keras.layers.Concatenate(name='feature_concat')([user_vec, movie_vec])
        
        # 多层神经网络
        dense = concat
        for i, units in enumerate(self.hidden_units):
            dense = keras.layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(1e-6),
                name=f'dense_{i+1}'
            )(dense)
            dense = keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(dense)
            dense = keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(dense)
          # 输出层
        if self.use_mixed_precision:
            # 混合精度模式下，输出层需要使用float32
            output = keras.layers.Dense(1, activation='linear', name='rating_output', dtype='float32')(dense)
        else:
            output = keras.layers.Dense(1, activation='linear', name='rating_output')(dense)
        
        # 创建模型
        model = keras.Model(inputs=[user_input, movie_input], outputs=output)
        
        # 编译模型
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.use_mixed_precision:
            # 混合精度训练需要使用LossScaleOptimizer
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
            
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("模型构建完成")
        
        return model
    
    def fit(self, movies_df: pd.DataFrame, epochs: int = 30, 
            min_interactions: int = 5) -> None:
        """
        训练深度学习推荐器
        
        Args:
            movies_df: 电影数据框
            epochs: 训练轮数
            min_interactions: 最小交互数
        """
        try:
            logger.info("开始训练深度学习推荐器...")
            
            # 准备交互数据
            interactions_df = self.prepare_interaction_data(movies_df, min_interactions)
            
            # 编码用户和电影ID
            interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
            interactions_df['movie_encoded'] = self.movie_encoder.fit_transform(interactions_df['movie_id'])
            
            # 构建模型
            num_users = len(self.user_encoder.classes_)
            num_movies = len(self.movie_encoder.classes_)
            self.build_model(num_users, num_movies)
            
            # 训练模型
            self.train(interactions_df, epochs=epochs, batch_size=128, validation_split=0.1)
            
            logger.info("深度学习推荐器训练完成")
            
        except Exception as e:
            logger.warning(f"深度学习推荐器训练失败: {e}")
            logger.info("将禁用深度学习功能")
            self.model = None
    
    def train(self, interactions_df: pd.DataFrame, epochs: int = 50, 
              batch_size: int = 256, validation_split: float = 0.2,
              early_stopping: bool = True, patience: int = 10) -> keras.callbacks.History:
        """
        训练模型
        
        Args:
            interactions_df: 交互数据
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            early_stopping: 是否使用早停
            patience: 早停耐心值
            
        Returns:
            训练历史
        """
        logger.info("开始训练深度学习模型...")
          # 准备训练数据
        X_users = interactions_df['user_encoded'].values
        X_movies = interactions_df['movie_encoded'].values
        y_ratings = np.array(interactions_df['rating'].values, dtype=np.float32)
        
        # 数据归一化
        min_rating = float(y_ratings.min())
        max_rating = float(y_ratings.max())
        y_ratings_normalized = (y_ratings - min_rating) / (max_rating - min_rating)
        
        # 分割数据
        X_users_train, X_users_val, X_movies_train, X_movies_val, y_train, y_val = train_test_split(
            X_users, X_movies, y_ratings_normalized, test_size=validation_split, random_state=42
        )
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )
        ]
        
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1
                )
            )
        
        # 训练模型
        history = self.model.fit(
            [X_users_train, X_movies_train], y_train,
            validation_data=([X_users_val, X_movies_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history
        logger.info("模型训练完成!")
        
        return history
    
    def predict_rating(self, user_id: str, movie_id: int) -> float:
        """
        预测用户对电影的评分
        
        Args:
            user_id: 用户ID
            movie_id: 电影ID
            
        Returns:
            预测评分
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        try:            # 编码用户和电影ID
            if user_id in self.user_encoder.classes_:
                user_encoded = int(self.user_encoder.transform([user_id])[0])
            else:
                # 对新用户使用平均嵌入
                user_encoded = 0
            
            if movie_id in self.movie_encoder.classes_:
                movie_encoded = int(self.movie_encoder.transform([movie_id])[0])
            else:
                # 对新电影返回平均评分
                return 5.0
            
            # 预测评分
            prediction = self.model.predict([
                np.array([user_encoded]), 
                np.array([movie_encoded])
            ], verbose=0)[0][0]
            
            # 反归一化并限制在合理范围内
            rating = prediction * 9 + 1  # 假设原始范围是1-10
            return np.clip(rating, 1.0, 10.0)
            
        except Exception as e:
            logger.warning(f"预测评分时出错: {e}")
            return 5.0
    
    def recommend_movies(self, user_id: str, movies_df: pd.DataFrame, 
                        n: int = 10, exclude_seen: bool = True) -> List[Dict]:
        """
        为用户推荐电影
        
        Args:
            user_id: 用户ID
            movies_df: 电影数据框
            n: 推荐数量
            exclude_seen: 是否排除已看过的电影
            
        Returns:
            推荐电影列表
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        recommendations = []
        
        for idx, movie in movies_df.iterrows():
            try:
                predicted_rating = self.predict_rating(user_id, movie['id'])
                
                recommendations.append({
                    'title': movie['title'],
                    'movie_id': movie['id'],
                    'predicted_rating': predicted_rating,
                    'actual_rating': movie.get('vote_average', 'N/A'),
                    'genres': movie.get('genres', []),
                    'release_date': movie.get('release_date', 'N/A'),
                    'popularity': movie.get('popularity', 0)
                })
            except Exception as e:
                logger.warning(f"为电影 {movie['title']} 生成推荐时出错: {e}")
                continue
        
        # 按预测评分排序
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n]
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.training_history is None:
            raise ValueError("没有训练历史可绘制")
        
        history = self.training_history.history
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(history['loss'], label='训练损失', color='blue')
        plt.plot(history['val_loss'], label='验证损失', color='red')
        plt.title('模型损失')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # MAE曲线
        plt.subplot(1, 3, 2)
        plt.plot(history['mae'], label='训练MAE', color='blue')
        plt.plot(history['val_mae'], label='验证MAE', color='red')
        plt.title('平均绝对误差')
        plt.xlabel('轮数')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线（如果可用）
        plt.subplot(1, 3, 3)
        if 'lr' in history:
            plt.plot(history['lr'], label='学习率', color='green')
            plt.title('学习率变化')
            plt.xlabel('轮数')
            plt.ylabel('学习率')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, '学习率数据不可用', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('学习率')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要信息"""
        if self.model is None:
            return {"status": "模型尚未构建"}
        
        summary = {
            "embedding_size": self.embedding_size,
            "hidden_units": self.hidden_units,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "total_params": self.model.count_params(),
            "num_users": len(self.user_encoder.classes_) if hasattr(self.user_encoder, 'classes_') else 0,
            "num_movies": len(self.movie_encoder.classes_) if hasattr(self.movie_encoder, 'classes_') else 0
        }
        
        if self.training_history:
            final_loss = self.training_history.history['val_loss'][-1]
            final_mae = self.training_history.history['val_mae'][-1]
            summary.update({
                "final_val_loss": final_loss,
                "final_val_mae": final_mae,
                "training_epochs": len(self.training_history.history['loss'])
            })
        
        return summary
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未构建")
        
        self.model.save(filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"模型已从 {filepath} 加载")
