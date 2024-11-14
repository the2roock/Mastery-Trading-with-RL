import os
import joblib

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch

from . import Cnn


class ClusteringModel(torch.nn.Module):
    def __init__(self, encoder: Cnn, pca: PCA, kmeans: KMeans):
        super().__init__()
        self.encoder = encoder
        self.flatten = torch.nn.Flatten()
        self.pca = pca
        self.kmeans = kmeans
    
    def forward(self, x):
        l1 = self.encoder(x)
        l1 = self.flatten(l1).cpu().detach().numpy()
        l2 = self.pca.transform(l1)
        out = self.kmeans.predict(l2)
        return out


def load_trained() -> ClusteringModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../static")
    cnn = torch.load(os.path.join(static_path, "CNN v1.pkl"), map_location=device)
    pca = joblib.load(os.path.join(static_path, "pca v1.pkl"))
    kmeans = joblib.load(os.path.join(static_path, "kmeans v1.pkl"))
    model = ClusteringModel(encoder=cnn, pca=pca, kmeans=kmeans)
    return model
