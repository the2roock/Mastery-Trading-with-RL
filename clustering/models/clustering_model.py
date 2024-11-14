from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from . import Cnn


class ClusteringModel:
    def __init__(self, encoder: Cnn, pca: PCA, kmeans: KMeans):
        self.encoder = encoder
        self.pca = pca
        self.kmeans = kmeans
    
    def forward(self, x):
        l1 = self.encoder(x)
        l2 = self.pca(l1)
        out = self.kmeans(l2)
        return out
