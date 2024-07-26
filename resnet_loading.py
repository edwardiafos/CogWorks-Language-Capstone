from cogworks_data.language import get_data_path
from pathlib import Path
import pickle


resnet18_features = {}

def load_resnet():
    global resnet18_features
    with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
        resnet18_features = pickle.load(f)

load_resnet()

print(len(resnet18_features))