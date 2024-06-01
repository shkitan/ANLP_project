import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def cluster(embeddings_array, k, threshold):
    """Two possible clustering algorithms:
    - if k is given, runs k means clustering algorithm
    - if threshold given, runs agglomerative clustering, that clusters according to the
        threshold (and we don't need to pre-determine the number of clusters).
    Since each sample is multi-label (can have more than 1 disease), it is hard to say
    how many clusters we want.
    As for knn algorithm, it is also not sure what the predicted label will be as
    neighbors are multi-labeled.
    """
    if k > -1 and threshold > 0:
        print("either k or threshold")
        return None

    if k > -1:
        clustering = KMeans(n_clusters=k)
    else:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)

    results = clustering.fit(embeddings_array)
    cluster_labels = results.labels_

    return cluster_labels



def eval_embeddings(embeddings_array, labels_array):
    """ evaluates the embeddings using three different intrinsic scores. """
    scores = {}
    scores["silhouette"] = silhouette_score(embeddings_array, labels_array)
    scores["calinski_harabasz"] = calinski_harabasz_score(embeddings_array, labels_array)
    scores["davies_bouldin"] = davies_bouldin_score(embeddings_array, labels_array)
    for result in scores:
        print(result, scores[result])


def visualize(embeddings_array, labels_array):
    """ project the data to 2d and visualize the original clustering in the
    high-dimension"""

    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(embeddings_array)

    # Visualize the clustering results
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels_array, cmap='viridis')
    plt.title('Clustering Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-k", type=int, help="number of clusters", default=-1)
    parser.add_argument("--threshold", type=float, help="threshold for neighbors", default=0)
    parser.add_argument("--embeddings_path", help="path to embeddings")
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings_path)
    labels = cluster(embeddings, args.k, args.threshold)

    eval_embeddings(embeddings, labels)
    visualize(embeddings, labels)


