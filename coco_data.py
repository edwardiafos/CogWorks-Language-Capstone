import json

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
            
        self.image_id_to_captions = self.image_to_caption()
        self.caption_id_to_images = self.caption_id_to_image()
        self.caption_id_to_captions = self.caption_id_to_caption()

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
