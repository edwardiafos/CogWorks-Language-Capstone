import string
from cogworks_data.language import get_data_path
from pathlib import Path
import json

def process_caption(caption):
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.lower()

    words = caption.split()

    return words



filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    coco_data = json.load(f)

caption_0 = process_caption(coco_data["annotations"][0]["caption"])
print(caption_0)