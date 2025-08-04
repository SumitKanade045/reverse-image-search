import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load saved feature vectors and image paths
features = np.load("features.npy")
paths = np.load("paths.npy", allow_pickle=True)

def search_image(query_feature, top_k=5):
    similarities = cosine_similarity([query_feature], features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(paths[i], float(similarities[i])) for i in top_indices]
