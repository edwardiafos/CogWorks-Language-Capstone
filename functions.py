import numpy as np
from pathlib import Path
import json
import pickle
from cogworks_data.language import get_data_path
from typing import List, Union, Sequence
from collections import Counter
from pathlib import Path
import os
from pathlib import Path
import json
from tokenizer import process_caption
from resnet_loading import load_resnet

from gensim.models import KeyedVectors
from coco_data import COCODataManager
from operator import itemgetter
from image2caption import Image2Caption

from train_model import train_model

### Load saved image descriptor vectors ###
'''resnet18_features = {}
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)'''


def initialize_coco_data():
    """
    coco_data can be used using the following:
    
    coco_data = initialize_coco_data()
    image_ids = coco_data.get_image_ids()
    captions = coco_data.get_caption_ids()
    image_to_captions = coco_data.image_to_caption()
    caption_id_to_images = coco_data.caption_id_to_image()
    caption_id_to_captions = coco_data.caption_id_to_caption()
    """
    filename = get_data_path("captions_train2014.json")
    coco_data = COCODataManager(filename)
    return coco_data



# poss need to modify to use resnet18 OR is that the step before when figuring out
# what image_descriptors get passed in?
def create_image_database(image_ids, image_descriptors, model):
    image_db = {}
    for img_id in image_ids:
        descriptor = image_descriptors.get(img_id)
        if descriptor is not None:
            embedding = Image2Caption(descriptor) 
            
            image_db[img_id] = embedding
    return image_db



def get_user_input():
    pass

def make_training_tuples():
    # training tuples should have:
    # (img_descriptor, semantic embedding of descriptor's caption, semantic embedding of diff img caption)
    # image ids come from resnet
    print("[Processing] loading resnet")
    resnet18_features = load_resnet()
    print(len(resnet18_features)) # e.g., 82612

    print(len(resnet18_features)) # 82612
    num_images_to_use = 500
    idxs = np.arange(len(resnet18_features))
    np.random.shuffle(idxs)

    selected_ids = [list(resnet18_features.keys())[key_idx] for key_idx in idxs[:num_images_to_use]]
    print(f"[Processing] Selected {len(selected_ids)} images")

    print("[Processing] Getting training descriptors")
    training_descriptor_vectors = np.asarray(itemgetter(*selected_ids)(resnet18_features))

    print("[Processing] Initializing Coco Data base")
    coco_data = initialize_coco_data()
    print("[Processing] Getting Caption IDs")
    caption_ids = [itemgetter(id)(coco_data.image_id_to_captions) for id in selected_ids]
    print("[Processing] Flattening Caption IDs")
    caption_ids = [item for sublist in caption_ids for item in sublist]
    print("[Processing] Getting Text Captions")
    text_captions = np.asarray([coco_data.caption_id_to_captions[caption_id] for caption_id in caption_ids])
    
    print("[Processing] Embedding captions")
    from concurrent.futures import ThreadPoolExecutor

    def process_chunk(captions_chunk):
        embeddings = {caption: se_text(caption, text_captions) for caption in captions_chunk}
        # Debugging: Check shapes
        for caption, embedding in embeddings.items():
            if embedding.shape != (200,):
                print(f"Warning: Chunk embedding shape for caption '{caption}' is {embedding.shape}, expected (200,)")
        return embeddings

    def process_captions_in_chunks(captions, chunk_size=100):
        """
        Without Jupyter notebook, the code run very slowly. This just processes functions simultaneously to make it run faste
        """
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, captions[i:i + chunk_size])
                    for i in range(0, len(captions), chunk_size)]
            results = [future.result() for future in futures]

        # Combine results from all chunks
        combined_results = {}
        for result in results:
            combined_results.update(result)

        return combined_results

    caption_to_embeddings = process_captions_in_chunks(text_captions)
    print(caption_to_embeddings)
    # To fix shape problems we had for 2 hours
    def get_embedding(caption):
        return caption_to_embeddings.get(caption, np.zeros(200))

    print("[Processing] Creating good embeddings")
    good_image_embeddings = np.array([get_embedding(caption) for caption in text_captions])
    print(good_image_embeddings)

    np.random.shuffle(text_captions)
    print("[Processing] Creating bad embeddings")
    bad_image_embeddings = np.array([get_embedding(caption) for caption in text_captions])
    print(bad_image_embeddings)
    
    # Print the shape of each embedding for debugging
    for i, embedding in enumerate(bad_image_embeddings):
        if embedding.shape != (200,):
            print(f"Warning: Embedding {i} shape is {embedding.shape}, expected (200,)")

    return training_descriptor_vectors, good_image_embeddings, bad_image_embeddings


def do_training():
    print("Training...")
    print("Making tuples")
    training_descriptor_vectors, good_image_embeddings, bad_image_embeddings = make_training_tuples()
    print("Getting accuracy")
    accuracy = train_model(training_descriptor_vectors, good_image_embeddings, bad_image_embeddings)
    print("GOT accuracy")

    print(accuracy)



def caption_embedder(captions):
    global glove 
    caption_tokens = [process_caption(cap) for cap in captions]

    vocab = set(caption_tokens)
    counters = []
    for caption_token in caption_tokens:
        counters.append(Counter(caption_token))

    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in caption_tokens]
    nt = np.array(nt, dtype=float)
    idf = np.log10(N / nt) # shape (N,)

    glove_embeddings = np.array([glove[word] for word in caption_tokens]) # shape (N, 200)
    for i, weight in enumerate(idf):
        glove_embeddings[i] += weight

    ret = glove_embeddings / np.linalg.norm(glove_embeddings) # shape (N, 200)
    return ret.mean(axis=0) # should be shape (200,)? hopefully??
   

def se_text(text: str, captions: Sequence[str]) -> np.ndarray: # um someone who has taken more math than algebra II please check this lol
    global glove, SE_Count
    
    #* Temp: Used for debugging
    SE_Count += 1
    print(f"se_text: {SE_Count}")


    text_tokens = process_caption(text)
    caption_tokens = [process_caption(cap) for cap in captions]

    total_tokens = [token for cap in captions for token in cap] + text_tokens
    vocab = set(total_tokens)

    counters = [Counter(tokens) for tokens in caption_tokens] + [Counter(text_tokens)]


    N = len(counters)
    nt = np.array([sum(1 if t in counter else 0 for counter in counters) for t in text_tokens], dtype=float)
    idf = np.log10(N / (nt + 1e-10))  #* Added a small constnt to avod division by zero


    glove_embeddings = []
    for word in text_tokens:
        if word in glove:
            glove_embeddings.append(glove[word])
        else:
            glove_embeddings.append(np.zeros(200))

    glove_embeddings = np.array(glove_embeddings)


    weighted_embeddings = glove_embeddings * idf[:, np.newaxis]

    norms = np.linalg.norm(weighted_embeddings, axis=1, keepdims=True)
    normalized_embeddings = weighted_embeddings / np.maximum(norms, 1e-10)

    return normalized_embeddings.mean(axis=0)


def match_caption_to_image(semantic_embedding_caption, semantic_embedding_images, k=4):
    """
    caption embedding (200,)
    all embedded images (N, 200) 
    returns k most similar image urls
    """
    print("Matching Captions")
    global coco_data
    res = np.dot(semantic_embedding_caption, semantic_embedding_images) # res = shape (N,) array
    res_sorted = np.argsort(res)[:k] # higher dot product, more similarity -- sorts indices
    top_image_embeddings = [semantic_embedding_images[i] for i in res_sorted] # shape (k,) -- gets top images based on sorted indices

    top_k_image_ids = [coco_data.se_image_to_image_id[se_image] for se_image in top_image_embeddings] #get using the image using database given top 4 semantic embeddings
    top_k_image_urls = [coco_data.image_id_to_urls[image_id] for image_id in top_k_image_ids]
    return top_k_image_urls


def quick_glove_loading(initialized=False, filename="glove.6B.200d.txt.w2v"): #! set initialized to True after running this one!!!
    from gensim.models import KeyedVectors
    if not initialized:
        print("Glove not initialized, Initializing...")

        glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

        # Save in bin to loadd
        glove.save("glove.6B.200d.bin")
        return glove
    if initialized:
        print("Glove alreeady initialized, Loading...")
        glove = KeyedVectors.load("glove.6B.200d.bin")
        return glove


if __name__ == '__main__':
    
    SE_Count = 0
    
    print("[Processing] GloVE word2vec keyed vectors")
    #! Remember to change initialized=True after running quick_glove_loading() once !!!
    glove = quick_glove_loading(initialized=False) # Will take longer at first, but become faster after running it first time
    print("[SUCCESS] GloVE word2vec keyed vectors")
    ### COCO DATABASE ###
    print("[Processing] Initializing Coco Database")
    coco_data = initialize_coco_data()
    print("[Success] Initializing Coco Database")
    print("[Processing] Training ResNet")
    do_training()
    print("[Success] Training ResNet")
