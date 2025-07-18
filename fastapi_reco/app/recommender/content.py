import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from sklearn.preprocessing import MinMaxScaler

class ContentRecommender:
    def __init__(self, csv_path="app/post_tags.csv"):
        df = pd.read_csv(csv_path).fillna("")
        df["tags"] = df["tags"].apply(lambda x: x.replace(",", " "))
        df["full_text"] = df["title"] + " " + df["content"] + " " + df["tags"]

        self.df = df
        self.post_ids = df["post_id"].values
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.X = self.vectorizer.fit_transform(df["full_text"])
        self.similarity_matrix = cosine_similarity(self.X)
        self.scaler = MinMaxScaler()

    def _get_common_tags(self, post_id: int, similar_post_id: int) -> List[str]:
        """두 게시물 간의 공통 태그를 찾습니다."""
        post_tags = set(self.df[self.df["post_id"] == post_id]["tags"].iloc[0].split())
        similar_post_tags = set(self.df[self.df["post_id"] == similar_post_id]["tags"].iloc[0].split())
        return list(post_tags.intersection(similar_post_tags))

    def _get_recommendation_reason(self, post_id: int, similar_post_id: int) -> str:
        """추천 이유를 생성합니다."""
        common_tags = self._get_common_tags(post_id, similar_post_id)
        if common_tags:
            return f"이 게시물은 다음 태그들을 공유합니다: {', '.join(common_tags[:3])}"
        return "이 게시물은 비슷한 주제를 다루고 있습니다."

    def recommend(self, post_id: int, top_k: int = 3) -> List[Dict[str, Any]]:
        if post_id not in self.post_ids:
            return []

        idx = list(self.post_ids).index(post_id)
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_k * 2]  # 더 많은 후보 생성

        # 유사도 점수 정규화
        scores = np.array([score for _, score in sim_scores])
        scores = self.scaler.fit_transform(scores.reshape(-1, 1)).flatten()

        results = []
        for (i, _), score in zip(sim_scores, scores):
            similar_post_id = int(self.df.iloc[i]["post_id"])
            reason = self._get_recommendation_reason(post_id, similar_post_id)
            
            results.append({
                "post_id": similar_post_id,
                "title": self.df.iloc[i]["title"],
                "similarity": round(float(score), 3),
                "reason": reason
            })

        # 상위 결과만 반환
        return results[:top_k]
