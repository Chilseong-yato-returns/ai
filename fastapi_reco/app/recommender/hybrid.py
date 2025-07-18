from recommender.als import ALSRecommender
from recommender.content import ContentRecommender


def hybrid_recommend(user_email, als: ALSRecommender, content: ContentRecommender, top_k=5):
    user_id = als.user_encoder.transform([user_email])[0]
    als_recs = als.recommend(user_id, top_k=top_k*2)

    seen_post_ids = set()
    final_recs = []

    for rec in als_recs:
        post_id = rec["post_id"]
        if post_id in seen_post_ids:
            continue

        content_recs = content.recommend(post_id, top_k=1)
        if content_recs:
            enriched_reason = content_recs[0]["reason"]
            rec["reason"] += f" ({enriched_reason})"
        
        final_recs.append(rec)
        seen_post_ids.add(post_id)
        if len(final_recs) >= top_k:
            break

    return final_recs
