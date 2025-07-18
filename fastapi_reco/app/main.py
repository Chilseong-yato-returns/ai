from fastapi import FastAPI, Query
from typing import List
from pydantic import BaseModel

from recommender.als import ALSRecommender
from recommender.content import ContentRecommender
from recommender.hybrid import hybrid_recommend

app = FastAPI()

als_model = ALSRecommender(csv_path="mock_data/user_post.csv")
content_model = ContentRecommender(csv_path="mock_data/post_tags.csv")

class RecommendationResult(BaseModel):
    post_id: int
    score: float
    reason: str

@app.get("/recommend", response_model=List[RecommendationResult])
def recommend(user_email: str = Query(...), top_k: int = Query(5)):
    try:
        results = hybrid_recommend(user_email, als_model, content_model, top_k=top_k)
        return results
    except Exception as e:
        return []
