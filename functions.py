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
from resnet_loading import resnet18_features

from gensim.models import KeyedVectors
from coco_data import COCODataManager
from operator import itemgetter

from train_model import train_model

filename = "glove.6B.200d.txt.w2v"
### this takes a while to load -- keep this in mind when designing your capstone project ###
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

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

### COCO DATABASE ###
coco_data = initialize_coco_data()


# poss need to modify to use resnet18 OR is that the step before when figuring out
# what image_descriptors get passed in?
def create_image_database(image_ids, image_descriptors, model):
    image_db = {}
    for img_id in image_ids:
        descriptor = image_descriptors.get(img_id)
        if descriptor is not None:
            embedding = None # TODO: After model is created, here, run the forward pass upon descriptor smth like model.forwardpass(descriptor)
            
            image_db[img_id] = embedding
    return image_db



def get_user_input():
    pass

def make_training_tuples():
    # training tuples should have:
    # (img_descriptor, semantic embedding of descriptor's caption, semantic embedding of diff img caption)
    # image ids come from resnet

    print(len(resnet18_features)) # 82612

    idxs = np.arange(len(resnet18_features))
    np.random.shuffle(idxs)

    print(len(idxs))

    training_idxs = idxs[0:len(resnet18_features)*3//4] # for even splitting purposes, 3/4 train, 1/4 test
    training_ids = [list(resnet18_features.keys())[key_idx] for key_idx in training_idxs]
    # ^ do the whole list(keys) thing bc dicts don't have indexes to access xyz elements,
    # so may need to make that a global variable to make sure the training/test don't 
    # overlap

    training_descriptor_vectors = np.asarray(itemgetter(*training_ids)(resnet18_features))

    #coco_data = initialize_coco_data()
    
    print(type(itemgetter(*training_ids)(coco_data.image_id_to_captions)), type(itemgetter(*training_ids)(coco_data.image_id_to_captions)[0]))
    caption_ids = np.asarray(itemgetter(*training_ids)(coco_data.image_id_to_captions), dtype = object)[:, 0]

    print(caption_ids.shape, caption_ids[:, 0].shape)

    cap_slice = caption_ids[:, 0]

    print(cap_slice)
    print(type(training_ids), type(caption_ids), type(coco_data.caption_id_to_captions))
    print(caption_ids.shape, caption_ids[:, 0].shape)
    print("wait")
    print(caption_ids[0][0])
    print("wait")
    print(len(caption_ids), len(caption_ids[0]))
    print(itemgetter(*caption_ids)(coco_data.caption_id_to_captions))
    text_captions = np.asarray(itemgetter(*caption_ids)(coco_data.caption_id_to_captions))
    caption_to_embeddings = {caption : se_text(caption, text_captions) for caption in text_captions}

    good_image_embeddings = np.asarray(itemgetter(*text_captions)(caption_to_embeddings))

    np.random.shuffle(text_captions) # so that we dont cut into testing data
    bad_image_embeddings = np.asarray(itemgetter(*text_captions)(caption_to_embeddings)) # now train idxs is idxs for bad img embeddings

    return training_descriptor_vectors, good_image_embeddings, bad_image_embeddings


def do_training():

    training_descriptor_vectors, good_image_embeddings, bad_image_embeddings = make_training_tuples()

    accuracy = train_model(training_descriptor_vectors, good_image_embeddings, bad_image_embeddings)

    print(accuracy)

do_training()


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
    """Takes text and returns a shape (200,) array by using IDF and glove embeddings.
    """
    global glove 
    text_tokens = process_caption(text) # len N
    caption_tokens = [process_caption(cap) for cap in captions]

    total_tokens = [token for cap in captions for token in process_caption(cap)] + text_tokens

    vocab = set(total_tokens)
    counters = []
    for caption_token in caption_tokens:
        counters.append(Counter(caption_token))
    counters.append(Counter(text_tokens))

    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in text_tokens]
    nt = np.array(nt, dtype=float)
    idf = np.log10(N / nt) # shape (N,)

    glove_embeddings = []
    for word in text_tokens:
        if word in glove:
            glove_embeddings.append(glove[word]) # append glove embedding if glove contains word, shape (200,)
        else:
            glove_embeddings.append(np.zeros(shape=(200,))) # else append array of zeros shape (200,)

    glove_embeddings = np.array(glove_embeddings) # shape (N, 200)
    for i, weight in enumerate(idf):
        glove_embeddings[i] += weight

    ret = glove_embeddings / np.linalg.norm(glove_embeddings) # shape (N, 200)
    return ret.mean(axis=0) # should be shape (200,)? hopefully??



def match_caption_to_image(semantic_embedding_caption, semantic_embedding_images, k=4):
    """
    caption embedding (200,)
    all embedded images (N, 200) 
    returns k most similar image urls
    """
    global coco_data
    res = np.dot(semantic_embedding_caption, semantic_embedding_images) # res = shape (N,) array
    res_sorted = np.argsort(res)[:k] # higher dot product, more similarity -- sorts indices
    top_image_embeddings = [semantic_embedding_images[i] for i in res_sorted] # shape (k,) -- gets top images based on sorted indices

    top_k_image_ids = [coco_data.se_image_to_image_id[se_image] for se_image in top_image_embeddings] #get using the image using database given top 4 semantic embeddings
    top_k_image_urls = [coco_data.image_id_to_urls[image_id] for image_id in top_k_image_ids]
    return top_k_image_urls
