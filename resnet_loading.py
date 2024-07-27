from cogworks_data.language import get_data_path
from pathlib import Path
import pickle




def load_resnet():
    resnet18_features = {}
    with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
        resnet18_features = pickle.load(f)
    return resnet18_features


if __name__ == '__main__':
    #* I modified the resnet to load non globally, bc I got errors in functions
    resnet18_features = load_resnet()
