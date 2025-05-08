import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, csv_path="app/post_tags.csv"):
        df = pd.read_csv(csv_path).fillna("")
        df["tags"] = df["tags"].apply(lambda x: x.replace(",", " "))
        df["full_text"] = df["title"] + " " + df["content"] + " " + df["tags"]

        self.df = df
        self.post_ids = df["post_id"].values
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(df["full_text"])
        self.similarity_matrix = cosine_similarity(self.X)

    def recommend(self, post_id: int, top_k: int = 3):
        if post_id not in self.post_ids:
            return []

        idx = list(self.post_ids).index(post_id)
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_k]

        return [
            {
                "post_id": int(self.df.iloc[i]["post_id"]),
                "title": self.df.iloc[i]["title"],
                "similarity": round(score, 3)
            }
            for i, score in sim_scores
        ]
