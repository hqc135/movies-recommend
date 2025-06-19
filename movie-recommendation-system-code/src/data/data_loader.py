"""
数据加载和预处理模块
负责加载数据、数据清洗和基础预处理
"""
import pandas as pd
import numpy as np
from ast import literal_eval
from typing import Tuple, List
import logging
import sys
import os

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.movies_df = None
        self.credits_df = None
        self.processed_df = None
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载原始数据文件"""
        try:
            self.config.validate_paths()
            logger.info("开始加载数据文件...")
            
            # 加载数据
            self.credits_df = pd.read_csv(self.config.CREDITS_FILE)
            self.movies_df = pd.read_csv(self.config.MOVIES_FILE)
            
            # 重命名列
            self.credits_df.columns = ['id', 'tittle', 'cast', 'crew']
            
            logger.info(f"成功加载电影数据: {len(self.movies_df)} 条记录")
            logger.info(f"成功加载演职人员数据: {len(self.credits_df)} 条记录")
            
            return self.movies_df, self.credits_df
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def merge_data(self) -> pd.DataFrame:
        """合并电影和演职人员数据"""
        if self.movies_df is None or self.credits_df is None:
            self.load_raw_data()
              # 合并数据
        merged_df = self.movies_df.merge(self.credits_df, on="id")
        logger.info(f"数据合并完成，共 {len(merged_df)} 条记录")
        
        return merged_df
    
    def clean_and_process_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """清洗和处理数据"""
        if df is None:
            df = self.merge_data()
            
        logger.info("开始数据清洗和预处理...")
        
        # 处理缺失值
        df["overview"] = df["overview"].fillna("")
          # 确保数值列是正确的数据类型
        numeric_columns = ['vote_count', 'vote_average', 'popularity', 'budget', 'revenue']
        for col in numeric_columns:
            if col in df.columns:
                # 将非数值转换为NaN，然后填充为0
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                logger.debug(f"列 {col}: {original_dtype} -> {df[col].dtype}, 空值数量: {df[col].isna().sum()}")
        
        # 额外的数据类型验证
        if 'vote_count' in df.columns:
            # 确保 vote_count 为非负数
            df['vote_count'] = df['vote_count'].clip(lower=0)
        
        if 'vote_average' in df.columns:
            # 确保 vote_average 在合理范围内 (0-10)
            df['vote_average'] = df['vote_average'].clip(lower=0, upper=10)
        
        # 处理release_date列
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        # 处理JSON格式的字段
        json_features = ["cast", "crew", "keywords", "genres"]
        for feature in json_features:
            if feature in df.columns:
                df[feature] = df[feature].apply(self._safe_literal_eval)
        
        # 提取导演信息
        if "crew" in df.columns:
            df["director"] = df["crew"].apply(self._get_director)
        
        # 提取前N个特征
        list_features = ["cast", "keywords", "genres"]
        for feature in list_features:
            if feature in df.columns:
                df[feature] = df[feature].apply(
                    lambda x: self._get_top_items(x, self.config.MAX_FEATURES_PER_CATEGORY)
                )
        
        # 清理文本数据
        text_features = ['cast', 'keywords', 'director', 'genres']
        for feature in text_features:
            if feature in df.columns:
                df[feature] = df[feature].apply(self._clean_text_data)
        
        # 创建组合特征
        df["soup"] = df.apply(self._create_soup, axis=1)
        
        # 计算IMDB加权评分
        df = self._calculate_weighted_rating(df)
        
        self.processed_df = df
        logger.info("数据预处理完成")
        
        return df
    
    def get_processed_data(self) -> pd.DataFrame:
        """获取处理后的数据"""
        if self.processed_df is None:
            return self.clean_and_process_data()
        return self.processed_df
    
    def _safe_literal_eval(self, x):
        """安全的literal_eval，处理异常情况"""
        try:
            return literal_eval(x) if pd.notna(x) else []
        except (ValueError, SyntaxError):
            return []
    
    def _get_director(self, crew_list: List) -> str:
        """从crew数据中提取导演信息"""
        if isinstance(crew_list, list):
            for person in crew_list:
                if isinstance(person, dict) and person.get("job") == "Director":
                    return person.get("name", "")
        return ""
    
    def _get_top_items(self, items_list: List, max_items: int = 3) -> List[str]:
        """提取前N个项目的名称"""
        if isinstance(items_list, list):
            names = [item.get("name", "") for item in items_list if isinstance(item, dict)]
            return names[:max_items]
        return []
    
    def _clean_text_data(self, x):
        """清理文本数据：转小写、去空格"""
        if isinstance(x, list):
            return [str(item).lower().replace(" ", "") for item in x if item]
        elif isinstance(x, str):
            return str(x).lower().replace(" ", "")
        else:
            return ""
    
    def _create_soup(self, row) -> str:
        """将所有特征组合成一个字符串"""
        features = []
        
        # 添加各种特征
        for feature in ['keywords', 'cast', 'genres']:
            if feature in row and isinstance(row[feature], list):
                features.extend(row[feature])
          # 添加导演
        if 'director' in row and row['director']:
            features.append(row['director'])
        
        return ' '.join(filter(None, features))
    
    def _calculate_weighted_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算IMDB加权评分"""
        try:
            # 确保数值列是数值类型
            df["vote_count"] = pd.to_numeric(df["vote_count"], errors='coerce').fillna(0)
            df["vote_average"] = pd.to_numeric(df["vote_average"], errors='coerce').fillna(0)
            
            # 计算平均评分和投票数阈值
            C = df["vote_average"].mean()
            m = df["vote_count"].quantile(self.config.VOTE_COUNT_QUANTILE)
            
            logger.info(f"计算加权评分: 平均评分={C:.3f}, 投票数阈值={m:.1f}")
            
            def weighted_rating(row):
                v = float(row["vote_count"])
                R = float(row["vote_average"])
                # 避免除零错误
                if v + m == 0:
                    return 0
                return (v/(v + m) * R) + (m/(v + m) * C)
            
            df["weighted_score"] = df.apply(weighted_rating, axis=1)
            
        except Exception as e:
            logger.error(f"计算加权评分时出错: {e}")
            # 如果出错，使用简单的平均评分作为加权分数
            df["weighted_score"] = pd.to_numeric(df["vote_average"], errors='coerce').fillna(0)
        
        return df
    
    def get_data_summary(self) -> dict:
        """获取数据摘要信息"""
        if self.processed_df is None:
            self.get_processed_data()
            
        summary = {
            "total_movies": len(self.processed_df),
            "total_genres": len(set().union(*self.processed_df["genres"].apply(lambda x: x if isinstance(x, list) else []))),
            "date_range": (
                self.processed_df["release_date"].min(),
                self.processed_df["release_date"].max()
            ),
            "average_rating": self.processed_df["vote_average"].mean(),
            "average_vote_count": self.processed_df["vote_count"].mean()
        }
        
        return summary
