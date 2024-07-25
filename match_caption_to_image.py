import numpy as np
def match_caption_to_image(semantic_embedding_caption, semantic_embedding_images):
    res = np.dot(semantic_embedding_caption, semantic_embedding_images)
    res_sorted = np.sort(res, 0)
    print(res_sorted[:4])

    return
    #get using the image using database given top 4 semantic embeddings
