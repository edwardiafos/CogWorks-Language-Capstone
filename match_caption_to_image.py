import numpy as np
def match_caption_to_image(semantic_embedding_caption, semantic_embedding_images, k=4):
    """
    caption embedding (200,)
    all embedded images (N,200) ?
    returns k most similar images
    """
    res = np.dot(semantic_embedding_caption, semantic_embedding_images) # (N,200)
    res_sorted = np.argsort(res, axis=0)
    print(semantic_embedding_images[:4])

    return
    #get using the image using database given top 4 semantic embeddings
