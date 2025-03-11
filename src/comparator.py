import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import adjusted_rand_score
from umap import UMAP


class RepresentationComparator:
    def __init__(self, rep1, rep2):
        """
        Initialize with two sets of representations to compare.

        Args:
            rep1 (np.ndarray): First set of representations
            rep2 (np.ndarray): Second set of representations
        """
        self.rep1 = rep1
        self.rep2 = rep2
        self.rep1_proc = None
        self.rep2_proc = None
        self.preprocess_representations()

    def preprocess_representations(self):
        """Normalize and align representations."""
        # Normalize
        self.rep1_proc = self.rep1 / np.linalg.norm(self.rep1, axis=1, keepdims=True)
        self.rep2_proc = self.rep2 / np.linalg.norm(self.rep2, axis=1, keepdims=True)

        # Align
        U, _, Vt = np.linalg.svd(self.rep2_proc.T @ self.rep1_proc)
        rotation = U @ Vt
        self.rep2_proc = self.rep2_proc @ rotation

    def linear_CKA(self):
        """Calculate Centered Kernel Alignment."""
        X = self.rep1_proc - self.rep1_proc.mean(0, keepdims=True)
        Y = self.rep2_proc - self.rep2_proc.mean(0, keepdims=True)

        XTX = X.T @ X
        YTY = Y.T @ Y
        XTY = X.T @ Y

        return (XTY @ XTY).trace() / np.sqrt((XTX @ XTX).trace() * (YTY @ YTY).trace())

    def correlation(self):
        """Calculate correlation between flattened representations."""
        return np.corrcoef(self.rep1_proc.flatten(), self.rep2_proc.flatten())[0, 1]

    def wasserstein(self):
        """Calculate average Wasserstein distance across dimensions."""
        distances = []
        for dim in range(self.rep1_proc.shape[1]):
            dist = wasserstein_distance(self.rep1_proc[:, dim], self.rep2_proc[:, dim])
            distances.append(dist)
        return np.mean(distances)

    def cluster_similarity(self, n_clusters=10):
        """Compare clustering results between representations."""
        kmeans1 = KMeans(n_clusters=n_clusters, random_state=42).fit(self.rep1_proc)
        kmeans2 = KMeans(n_clusters=n_clusters, random_state=42).fit(self.rep2_proc)
        return adjusted_rand_score(kmeans1.labels_, kmeans2.labels_)

    def dimensionality_reduction(self, method="tsne"):
        """
        Perform dimensionality reduction using specified method.

        Args:
            method (str): One of 'tsne', 'umap', or 'pca'
        """
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "umap":
            reducer = UMAP(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be one of 'tsne', 'umap', or 'pca'")

        rep1_reduced = reducer.fit_transform(self.rep1_proc)
        rep2_reduced = reducer.fit_transform(self.rep2_proc)

        return rep1_reduced, rep2_reduced

    def visualize(self, save_path=None, add_lines=False):
        """Create visualization plots for all reduction methods."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot for each dimensionality reduction method
        for idx, method in enumerate(["tsne", "umap", "pca"]):
            rep1_reduced, rep2_reduced = self.dimensionality_reduction(method)

            axes[idx].scatter(
                rep1_reduced[:, 0], rep1_reduced[:, 1], alpha=0.5, label="Rep1"
            )
            axes[idx].scatter(
                rep2_reduced[:, 0], rep2_reduced[:, 1], alpha=0.5, label="Rep2"
            )

            # Add lines between corresponding points
            if add_lines:
                for i in range(min(rep1_reduced.shape[0], rep2_reduced.shape[0])):
                    axes[idx].plot(
                        [rep1_reduced[i, 0], rep2_reduced[i, 0]],
                        [rep1_reduced[i, 1], rep2_reduced[i, 1]],
                        color="black",
                        alpha=0.2,
                    )
            axes[idx].set_title(method.upper())
            axes[idx].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def compare(self):
        """Run all comparison metrics and return results."""
        results = {
            "cka": self.linear_CKA(),
            "correlation": self.correlation(),
            "wasserstein": self.wasserstein(),
            "cluster_similarity": self.cluster_similarity(),
        }
        return results

    def full_analysis(self, save_plot=None, add_lines=False):
        """Perform full analysis with metrics and visualization."""
        metrics = self.compare()
        self.visualize(save_path=save_plot, add_lines=add_lines)
        return metrics
