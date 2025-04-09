from fastapi import FastAPI, Query
from app.recommender_content import ContentRecommender
from app.recommender_als import ALSRecommender
from app.logger import save_log

app = FastAPI()  # ✅ 반드시 필요!

content_model = ContentRecommender()
als_model = ALSRecommender()

@app.get("/recommend/hybrid")
def hybrid_recommend(user_id: int = Query(...), post_id: int = Query(...), top_k: int = 5):
    als_result = als_model.recommend(user_id, top_k)
    content_result = content_model.recommend(post_id, top_k)

    result = {
        "recommendations": {
            "for_you": als_result,
            "similar_to_this": content_result
        }
    }

    save_log({
        "user_id": user_id,
        "post_id": post_id,
        "top_k": top_k,
        "result": result
    })

    return result
