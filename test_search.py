import numpy as np
from image_search import search_image

# Load a sample query feature (for testing, reuse the first one from features.npy)
features = np.load("features.npy")
query_feature = features[0]

# Search for similar images
results = search_image(query_feature)

# Print results
for path, score in results:
    print(f"{path}: {score}")
