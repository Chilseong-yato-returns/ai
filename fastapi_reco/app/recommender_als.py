import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

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

        matrix = csr_matrix((
            df["prefer"].astype(np.float32),
            (df["item_id"], df["user_id"])
        ))

        self.model = AlternatingLeastSquares(
            factors=20,
            regularization=0.1,
            iterations=50,
            use_gpu=False
        )
        self.model.fit(matrix)
        self.user_item_matrix = matrix.T.tocsr()

    def recommend(self, user_id: int, top_k: int = 5):
        if user_id not in self.user_ids:
            return []

        user_items = self.user_item_matrix[user_id]
        num_seen = user_items.getnnz()
        num_total = self.user_item_matrix.shape[1]
        num_unseen = num_total - num_seen

        if num_unseen <= 0:
            return []

        safe_k = min(top_k, num_unseen)

        try:
            item_ids, scores = self.model.recommend(
                user_id,
                user_items,
                N=safe_k,
                filter_already_liked_items=True
            )
        except Exception as e:
            print(f"⚠️ ALS 추천 오류: {e}")
            return []

        results = []
        for item_id, score in zip(item_ids, scores):
            if np.isnan(score) or score < -1e+30:
                continue
            post_id = self.item_encoder.inverse_transform([item_id])[0]
            results.append({
                "post_id": int(post_id),
                "score": round(float(score), 3)
            })

        return results
