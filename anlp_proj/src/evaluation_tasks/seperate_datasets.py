
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import os
from argparse import ArgumentParser

PROJECT_ROOT = '/cs/labs/tomhope/guy_korn/ANLP_PROJECT/'
def present_projection(embeddings_array, labels, outpath, predictions=None):
    pca = PCA(n_components=2)

    x_2d = pca.fit_transform(embeddings_array)

    if predictions is not None:
        # present predictions with shape as class pred and color as label
        predictions_one = np.where(predictions == 1)[0]
        plt.scatter(x_2d[predictions_one, 0], x_2d[predictions_one, 1], c=labels[predictions_one].tolist(),
                    cmap='viridis', shape="*")
        predictions_zero = np.where(predictions == 0)[0]
        plt.scatter(x_2d[predictions_zero, 0], x_2d[predictions_zero, 1], c=labels[predictions_zero].tolist(),
                    cmap='viridis', shape="+")
        title = 'Clustering Visualization with predictions'
    else:
        # Visualize the clustering results
        plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, cmap='viridis')
        title = 'Clustering Visualization'
    plt.colorbar(label='Class')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(outpath)


def prepare_data(obs_ds, smokers_ds):
    embeddings_array = np.concatenate((obs_ds, smokers_ds))
    labels = np.concatenate(
        (np.ones((obs_ds.shape[0],)), np.zeros((smokers_ds.shape[0],))))

    return embeddings_array, labels


def evaluate_separation(X_train, y_train, X_test, y_test, out_path, k=10):
    # Create a KNN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    with open(out_path, 'wt') as f:
        print(f"{k}-nearest neighbors accuracy:", accuracy, file=f)

    return y_pred


def load_data(dir_path):
    x = []
    for name in os.listdir(dir_path):
        arr = np.load(os.path.join(dir_path, name))
        x.append(arr)
    return np.array(x)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--obs_train")
    parser.add_argument("--smokers_train")
    parser.add_argument("--obs_test")
    parser.add_argument("--smokers_test")
    parser.add_argument("--section_split")
    args = parser.parse_args()
    x_obs_train = load_data(os.path.join(PROJECT_ROOT, args.obs_train))
    x_obs_test = load_data(os.path.join(PROJECT_ROOT, args.obs_test))
    x_obs = np.concatenate((x_obs_train, x_obs_test))
    x_smokers_train = load_data(os.path.join(PROJECT_ROOT, args.smokers_train))
    x_smokers_test = load_data(os.path.join(PROJECT_ROOT, args.smokers_test))
    x_smokers = np.concatenate((x_smokers_train, x_smokers_test))
    all_data = prepare_data(x_obs, x_smokers)  # concatenate all data for pca

    splited_path = args.obs_train.split(os.sep)

    out_dir = os.path.join(PROJECT_ROOT, "results", "separate_datasets", args.section_split, *splited_path[3:])
    print(f"Save result at {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    present_projection(*all_data, outpath=os.path.join(out_dir, "pca.png"))

    # load train/test for knn
    train_data = prepare_data(x_obs_train, x_smokers_train)
    test_data = prepare_data(x_obs_test, x_smokers_test)
    predictions_y = evaluate_separation(*train_data, *test_data, out_path=os.path.join(out_dir, "knn-acc.txt"))
    # present_projection(*test_data, outpath=os.path.join(out_dir, "predictions.png"), predictions=predictions_y)

