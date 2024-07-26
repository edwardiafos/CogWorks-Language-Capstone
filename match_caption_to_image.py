import numpy as np
def match_caption_to_image(semantic_embedding_caption, semantic_embedding_images, k=4):
    """
    caption embedding (1,200)
    all embedded images (N,200) ?
    returns k most similar images
    """
    res = np.dot(semantic_embedding_caption, semantic_embedding_images) # (N,1)
    res_sorted = np.argsort(res, axis=0)
    top_image_embeddings = semantic_embedding_images[:k]

    #return
    #get using the image using database given top 4 semantic embeddings
