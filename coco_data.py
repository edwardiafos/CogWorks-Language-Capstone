import json
from pathlib import Path
from cogworks_data.language import get_data_path
import pickle
import numpy as np

from ___ import image2caption_model # import the trained model from somewhere

### Load saved image descriptor vectors ###
resnet18_features = {}
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)

class COCODataManager:
    """
    Takes in smth like this 'captions_train2014.json' and can be called to get:
    image to caption
    caption to image
    caption to caption
    image ids
    caption ids
    """
    def __init__(self, captions_file):
        with open(captions_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.image_id_to_captions = self.image_to_caption() # returns dict of image ids mapped to its associated caption ids
        self.caption_id_to_images = self.caption_id_to_image() # returns dict of caption ids mapped to image ids
        self.caption_id_to_captions = self.caption_id_to_caption() # returns dict of caption ids mapped to text captions
        self.se_image_to_image_ids = self.se_image_to_image_id() # returns dict of image semantic embeddings mapped to their image ids
        
    def image_to_caption(self):
        dict = {}
        for annotation in self.coco_data["annotations"]:
            img_id = annotation["image_id"]
            cap_id = annotation["id"]
            if img_id not in dict:
                dict[img_id] = []
            dict[img_id].append(cap_id)
        return dict

    def caption_id_to_image(self):
        dict = {}
        for annotation in self.coco_data["annotations"]:
            dict[annotation["id"]] = annotation["image_id"]
        return dict

    def caption_id_to_caption(self):
        dict = {}
        for annotation in self.coco_data["annotations"]:
            dict[annotation["id"]] = annotation["caption"]
        return dict

    def get_image_ids(self):
        return [img["id"] for img in self.coco_data["images"]]

    def get_caption_ids(self):
        return list(self.caption_id_to_caption.keys())
    
    def get_image_urls(self):
        return [img["url"] for img in self.coco_data["coco_url"]]

    def se_image_to_image_id(self):
        ret = {}
        for image_id, descriptor_vector in resnet18_features.items():
            # image2caption_model takes in an input shape (N, 512) and output should be a shape (N, 50)
            ret[image2caption_model(descriptor_vector[np.newaxis, :])[0]] = image_id   # this line makes it so that the keys of the dict are shape (50,) !! keep this in mind

        return ret
