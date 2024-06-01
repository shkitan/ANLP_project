import json
import os
import numpy as np
import pandas as pd
import random
# from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from argparse import ArgumentParser

PROJECT_ROOT = '/cs/labs/tomhope/guy_korn/ANLP_PROJECT/'
#
# def multi(x_train, y_train, x_test, y_test, out_dir):
#     classifiers = {"RF": ClassifierChain(RandomForestClassifier()),
#                    "MLP": ClassifierChain(MLPClassifier()),
#                    "logistic": ClassifierChain(LogisticRegression())}
#     results = pd.DataFrame()
#     for classifier_name in classifiers:
#         classifier = classifiers[classifier_name]
#         # Training logistic regression model on train data
#         classifier.fit(x_train, y_train)
#         # predict
#         predictions = classifier.predict(x_test)
#         # accuracy
#         results = results.append({"base classifier": classifier_name,
#                                   "accuracy": accuracy_score(y_test, predictions)},
#                                  ignore_index=True)
#
#     out_path = os.path.join(out_dir, "chain_results.csv")
#     results.to_csv(out_path)


def ons_vs_rest(x_train, y_train, x_test, y_test, out_dir):
    pipelines = {"RF": Pipeline([('clf', OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1)),]),
                 "MLP": Pipeline([('clf', OneVsRestClassifier(MLPClassifier(max_iter=1000), n_jobs=-1)),]),
                 "logistic": Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=2000), n_jobs=-1)),])}
    results = pd.DataFrame(columns=["base classifier", "micro precision",
                                    "micro recall", "macro precision", "macro recall"])

    for pipeline_name in pipelines:
        pipeline = pipelines[pipeline_name]
        tps, fps, fns, precisions, recalls = [], [], [], [], []
        for index in range(y_train.shape[1]):
            pipeline.fit(x_train, y_train[:, index])
            # calculating test accuracy
            predictions = pipeline.predict(x_test)
            confusion = confusion_matrix(y_test[:, index], predictions)
            tps.append(confusion[1, 1])
            fps.append(confusion[0, 1])
            fns.append(confusion[1, 0])
            if np.sum(confusion[:, 1]) == 0:
                precisions.append(0)
            else:
                precisions.append(confusion[1, 1] / np.sum(confusion[:, 1]))
            recalls.append(confusion[1, 1] / np.sum(confusion[1, :]))

        sum_tps = sum(tps)
        sum_all_pred_pos = sum([tps[index] + fps[index] for index in range(len(tps))])
        sum_all_pos = sum([tps[index] + fns[index] for index in range(len(tps))])
        micro_precision = sum_tps / sum_all_pred_pos
        micro_recall = sum_tps / sum_all_pos
        results.loc[len(results)] = {"base classifier": pipeline_name,
                                  "micro precision": micro_precision,
                                  "micro recall": micro_recall,
                                  "macro precision": np.mean(precisions),
                                  "macro recall": np.mean(recalls)}

    out_path = os.path.join(out_dir, "ons_vs_rest.csv")
    results.to_csv(out_path, index=False)
    print(results.to_markdown(index=False))


def generate_random_ds():

    # Define the labels and features
    num_samples = 1000  # Number of samples in the dataset
    num_labels_per_sample = 2  # Number of labels per sample
    num_features = 20  # Number of features for each sample
    # Define the length of the vector and number of '1's
    vector_length = 4  # Length of the vector

    # Generate random multi-labeled dataset
    x, y = [], []
    for _ in range(num_samples):
        features = [random.random() for _ in range(num_features)]
        num_ones = random.sample([_ for _ in range(vector_length)], k=1)[0]  # Number of '1's in the vector
        # Generate a random multi-hot vector
        labels = np.zeros(vector_length, dtype=int)
        positions = np.random.choice(vector_length, num_ones, replace=False)
        labels[positions] = 1
        # selected_labels = random.sample(labels, num_labels_per_sample)
        x.append(features)
        y.append(labels)

    x = np.array(x)
    y = np.array(y)
    x_train, y_train = x[:int(num_samples*0.8)], y[:int(num_samples*0.8)]
    x_test, y_test = x[int(num_samples*0.8):], y[int(num_samples*0.8):]

    return x_train, y_train, x_test, y_test


def load_data(dir_path, labels):
    x, y = [], []
    for name in os.listdir(dir_path):
        patient_id = name.replace(".npy", "")
        if patient_id not in labels:
            print(f"{patient_id} not in labels")
            continue
        arr = np.load(os.path.join(dir_path, name))
        x.append(arr)
        y.append(labels[patient_id])
    return np.array(x), np.array(y)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--embeddings_train")
    parser.add_argument("--embeddings_test")
    parser.add_argument("--section_split")
    args = parser.parse_args()
    emb_train = os.path.join(PROJECT_ROOT, args.embeddings_train)
    split_path = args.embeddings_train.split(os.sep)
    emb_test = os.path.join(PROJECT_ROOT, args.embeddings_test)

    with open(os.path.join(PROJECT_ROOT, "datasets/obesity/disease_tag.json"), 'rt') as f:
        obs_labels = json.load(f)
    x_train, y_train = load_data(emb_train, obs_labels)
    x_test, y_test = load_data(emb_test, obs_labels)
    results_path = os.path.join(PROJECT_ROOT, "results", "comorbidities", args.section_split, *split_path[3:])
    print(f"Saving in {results_path}")
    os.makedirs(results_path, exist_ok=True)
    # ds = generate_random_ds()
    # multi(*ds, "")
    ons_vs_rest(x_train, y_train, x_test, y_test, results_path)


