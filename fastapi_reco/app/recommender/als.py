import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from typing import List, Dict, Any

class ALSRecommender:
    def __init__(self, csv_path="app/user_post.csv"):
        df = pd.read_csv(csv_path)
        df["prefer"] = df["prefer"].fillna(1).clip(lower=1)

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        df["user_id"] = self.user_encoder.fit_transform(df["user_email"])
        df["item_id"] = self.item_encoder.fit_transform(df["post_id"])

        self.df = df
        self.user_ids = df["user_id"].unique()
        self.scaler = MinMaxScaler()

        matrix = csr_matrix((
            df["prefer"].astype(np.float32),
            (df["item_id"], df["user_id"])
        ))

        self.model = AlternatingLeastSquares(
            factors=32,  # 잠재 요인 수 증가
            regularization=0.1,
            iterations=50,
            use_gpu=False
        )
        self.model.fit(matrix)
        self.user_item_matrix = matrix.T.tocsr()

    def _get_recommendation_reason(self, user_id: int, item_id: int) -> str:
        """추천 이유를 생성합니다."""
        user_items = self.user_item_matrix[user_id]
        similar_items = self.model.similar_items(item_id, N=3)
        
        if len(similar_items) > 0:
            return f"이 게시물은 당신이 관심을 보인 다른 게시물들과 유사합니다."
        return "이 게시물은 당신의 관심사와 잘 맞습니다."

    def _ensure_diversity(self, recommendations: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """추천 결과의 다양성을 보장합니다."""
        if len(recommendations) <= top_k:
            return recommendations

        # 점수 기반 정렬
        sorted_recs = sorted(recommendations, key=lambda x: x["score"], reverse=True)
        
        # 상위 결과는 유지하고 나머지에서 다양하게 선택
        diverse_recs = sorted_recs[:top_k//2]
        remaining = sorted_recs[top_k//2:]
        
        # 나머지에서 랜덤하게 선택
        if remaining:
            diverse_recs.extend(np.random.choice(remaining, 
                                               size=min(len(remaining), top_k - len(diverse_recs)), 
                                               replace=False))
        
        return diverse_recs

    def recommend(self, user_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        if user_id not in self.user_ids:
            return []

        user_items = self.user_item_matrix[user_id]
        num_seen = user_items.getnnz()
        num_total = self.user_item_matrix.shape[1]
        num_unseen = num_total - num_seen

        if num_unseen <= 0:
            return []

        safe_k = min(top_k * 2, num_unseen)  # 더 많은 후보를 생성

        try:
            item_ids, scores = self.model.recommend(
                user_id,
                user_items,
                N=safe_k,
                filter_already_liked_items=True
            )
            
            # 점수 정규화
            scores = self.scaler.fit_transform(scores.reshape(-1, 1)).flatten()
            
            results = []
            for item_id, score in zip(item_ids, scores):
                if np.isnan(score) or score < 0:
                    continue
                    
                post_id = self.item_encoder.inverse_transform([item_id])[0]
                reason = self._get_recommendation_reason(user_id, item_id)
                
                results.append({
                    "post_id": int(post_id),
                    "score": round(float(score), 3),
                    "reason": reason
                })

            # 다양성 보장
            results = self._ensure_diversity(results, top_k)
            
            return results

        except Exception as e:
            print(f"⚠️ ALS 추천 오류: {e}")
            return []
