import argparse
import os
from pathlib import Path
from enum import Enum

import torch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
#import seaborn as sns
#import wandb

from load_data import get_records_by_patient, get_patients_dict
from section_embedding import embed_patient_sections

random.seed(1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


#### Triplet Loss and the Triplets Dataset classes
all_mean_distance_pos = []
all_mean_distance_neg = []
all_mean_org_distance = []

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN, positive_weight=1.0, org_dist_weight=0.9, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.positive_weight = positive_weight
        self.org_dist_weight = org_dist_weight

    def forward(self, anchor, positive, negative, org_anchor, org_positive, org_negative, all_distance_pos, all_distance_neg, all_org_distance):
        # Calculate the distance between the anchor and positive example, and between the anchor and negative example
        distance_pos = self.distance_metric(anchor, positive) # type: ignore
        distance_neg = self.distance_metric(anchor, negative) # type: ignore
        
        # Calculate the distance between the each embeedings and the original embeddings
        org_distance = 1 - F.cosine_similarity(anchor, org_anchor, dim=-1) + 1 - F.cosine_similarity(org_positive, positive, dim=-1) + 1 - F.cosine_similarity(org_negative, negative, dim=-1)

        # Save the distances for plotting
        all_distance_pos.append(distance_pos.clone().detach())
        all_distance_neg.append(distance_neg.clone().detach())
        all_org_distance.append(org_distance.clone().detach())

        # Calculate the loss using the distance metric for the positive and negative examples, and using cosine similarity for the original embeddings
        loss = F.relu(self.positive_weight * distance_pos - distance_neg + self.margin)
        org_loss = F.relu(org_distance)

        # Combine the losses to one loss
        loss = loss + self.org_dist_weight * org_loss

        return loss.mean()


class TripletDatasetSection(torch.utils.data.Dataset):
    ''' A dataset that holds the triplets of sections.'''
    def __init__(self, patient_data, emb_dir, section_are_max_seq_len=False):
        # self.patient_data = self.preprocess_patient_data(patient_data)
        self.patient_data = self.pre_process_patient_data(patient_data,emb_dir,section_are_max_seq_len)
        self.section_ids_dict = self.build_section_ids_dict()
        self.triplet_keys = list(self.form_section_triplets())
        self.emb_dir = emb_dir
        

    def pre_process_patient_data(self, patient_data, emb_dir, section_are_max_seq_len):
      # if the section were created with max_seq_len, we need to change patient_sections keys to be section numbers according to tokenizer
      if section_are_max_seq_len:
        new_patient_data = {}
        for patient_id in patient_data.keys():
            new_patient_data[patient_id] = {}
            # Load embedding for patient_id
            num_of_sections = np.load(f"{emb_dir}/{patient_id}.npy").shape[0]
            # Create a dict of section_id to section_number with text which is ""
            for section_num in range(1,num_of_sections+1):
                new_patient_data[patient_id][f"section{section_num}-{section_num}"] = ""
        
        # print a random patient to see the new_patient_data
        print(f"new_patient_data={new_patient_data[random.choice(list(new_patient_data.keys()))]}")

        return new_patient_data
      else:
        return patient_data
    
    def build_section_ids_dict(self):
      ''' Builds a dictionary that keeps a list of patients with sections that could be used as positive examples.
      A patient's section could be used as a postivie example if it has the same section_id, and it's not the same section.
        The keys are the section_ids and the values are lists of patient_index which hold these keys.
        Example for the self.patient_data kets is -
        self.patient_data = {1:{("fam",1):"bla1",("dig",2):"bla2"},2:{("fam",1):"goo1",("dig",2):"goo2"},3:{("fami",1):"foo1",("dig",2):"foo2"}}
        then the dict would hold a key ("fam",1) with the value [1,2], and so on.
      '''

      section_ids_dict = {}
      for patient_id, patient_sections in self.patient_data.items():
          for section_id in patient_sections.keys():
              # If section has not been seen before
              if section_id not in section_ids_dict:
                  section_ids_dict[section_id] = []
              # Add this patient_id to the list of possible positive examples for this section_id
              section_ids_dict[section_id].append(patient_id)
      # print random section_id to see the section_ids_dict
      return section_ids_dict

    def form_section_triplets(self):
      # Form up the triplets of tuples identifying the sections
      for id_patient, patient_sections in self.patient_data.items():
          for anchor_key in patient_sections.keys():
              positive_patient_index, positive_key = self.get_positive_sample(anchor_key, id_patient)
              negative_patient_index, negative_key = self.get_negative_sample(anchor_key, id_patient)

              yield id_patient, anchor_key, positive_patient_index, positive_key, negative_patient_index, negative_key
      

    def get_positive_sample(self, anchor_key, anchor_patient_id):
      '''
       For the given section, find a random positive example using the list of potential patients.
       If a section values list has only one patient, it means this is a unique section and we should return the anchor key.
      '''
      positive_patient_ids = self.section_ids_dict.get(anchor_key, [])
      
      # If no other patient shares this section, return anchor_key as a positive example
      if len(positive_patient_ids) == 1 and positive_patient_ids[0] == anchor_patient_id:
          return anchor_patient_id, anchor_key

      positive_patient_ids = [pat_id for pat_id in positive_patient_ids if pat_id != anchor_patient_id]
      
      if not positive_patient_ids:  # If the list is empty, it means there is no other patient with the same section.
          raise ValueError('Cannot find a positive sample. Please check the section distribution.')

      positive_patient_id = random.choice(positive_patient_ids)

      return positive_patient_id, anchor_key
    
    def get_negative_sample(self, anchor_key, anchor_patient_index):
        while True:
            negative_patient_index = random.choice([i for i in list(self.patient_data.keys()) if i != anchor_patient_index])
            negative_patient_keys = self.patient_data[negative_patient_index].keys()
            list_of_possible_negative_keys = [key for key in negative_patient_keys if key != anchor_key]
            if not list_of_possible_negative_keys:
                continue
            negative_key = random.choice(list_of_possible_negative_keys)
            if negative_key is not None and not isinstance(negative_key, int):  # Make sure we selected a section key
                return negative_patient_index, negative_key

    def __getitem__(self, idx):
        anchor_patient_idx, anchor_key, positive_patient_idx, positive_key, negative_patient_idx, negative_key = \
            self.triplet_keys[idx]

        anchor_section = self.patient_data[anchor_patient_idx][anchor_key]
        positive_section = self.patient_data[positive_patient_idx][positive_key]
        negative_section = self.patient_data[negative_patient_idx][negative_key]

        return ({"pat_id": anchor_patient_idx, "section_id": anchor_key, "section_text": anchor_section},
                {"pat_id": positive_patient_idx, "section_id": positive_key, "section_text": positive_section},
                {"pat_id": negative_patient_idx, "section_id": negative_key, "section_text": negative_section})

    def __len__(self):
        return len(self.triplet_keys)


#### Training Functions
def load_embedding(section, emb_dir):
    ''' Loads the embedding for the given section. '''
    patient_id = section['pat_id'][0]
    file_path = f"{emb_dir}/{patient_id}.npy"
    try:
        emb = np.load(file_path)
    except FileNotFoundError:
        print(f"Could not find embedding for patient {patient_id}")
        raise ValueError

    return emb


def load_all_embeddings_to_dict(dataloader, input_emb_dir):
    ''' Loads all the embeddings to a dictionary. '''
    embeddings = {}
    for anchor, positive, negative in dataloader:
        for sec in [anchor, positive, negative]:
            if sec['pat_id'][0] not in embeddings:
                emb = load_embedding(sec, input_emb_dir)
                if np.isnan(emb).any():
                    print(f"load_all_embeddings_to_dict:Found nan in embedding for patient {sec['pat_id'][0]}")
                    raise ValueError
                embeddings[sec['pat_id'][0]] = torch.tensor(emb,
                                                                dtype=torch.float,
                                                                requires_grad=True,
                                                                device=device)

    return embeddings


def get_section_embedding(patient_embeddings, section_id):
    try:
        section_index = int(section_id[0].split("-")[1])
    except Exception as e:
        print(f"Could not parse section index from {section_id[0]}")
        raise e

    if torch.isnan(patient_embeddings).any():
        print("get_section_embedding:Found nan in original patient embeddings,before slicing")
        raise ValueError
    
    try:
        return patient_embeddings[section_index - 1]
    except Exception as e:
        print(f"Could not get section embedding for section {section_id[0]}")
        raise e
        # A temp hack - We don't know the section index, so we return the last section embedding
        # return patient_embeddings[-1]

def plot_distances(all_mean_distance_pos, all_mean_distance_neg, all_mean_org_distance, output_dir):
    ''' Plots the distances during training. It saves the plot to a pdf file. '''
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})

    plt.plot(all_mean_distance_pos, label="Positive", color='green', linewidth=3, marker='o', markerfacecolor='green', markersize=10)
    plt.plot(all_mean_distance_neg, label="Negative", color='red', linewidth=3, marker='o', markerfacecolor='red', markersize=10)
    plt.plot(all_mean_org_distance, label="Original", color='blue', linewidth=3, marker='o', markerfacecolor='blue', markersize=10)

    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.legend()
    plt.xticks(range(len(all_mean_distance_pos)))
    
    # Save plot to pdf
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    plt.savefig(f"{output_dir}/figures/distances.pdf")



def train_embedding_model(patient_data: dict, input_emb_dir: str, output_dir: str, epochs=10, lr=1e-3, distance_metric=TripletDistanceMetric.EUCLIDEAN, margin=2.0, positive_weight=1.0, org_dist_weight=0.5, section_are_max_seq_len=False):
    #wandb.init(project="anlp_triplet_loss", name=f"{input_emb_dir}")  # Initialize Weights and Biases
    dataset = TripletDatasetSection(patient_data, input_emb_dir, section_are_max_seq_len)
    dataloader = DataLoader(dataset, shuffle=True)

    # Init lists for saving the mean distances
    all_distance_pos = []
    all_distance_neg = []
    all_org_distance = []

    # Load all embeddings before training
    embeddings = load_all_embeddings_to_dict(dataloader, input_emb_dir)
    original_embeddings = embeddings.copy()

    criterion = TripletLoss(distance_metric=distance_metric, margin=margin, positive_weight=positive_weight, org_dist_weight=org_dist_weight)
    optimizer = torch.optim.Adam(embeddings.values(), lr=lr)

    # uninformative section embeddings
    #num_of_sections_with_zero_loss = 0

    for epoch in range(epochs):
        for anchor, positive, negative in tqdm(dataloader):
            # Extracting pat_id values of the triplets
            pat_ids = [anchor['pat_id'][0], positive['pat_id'][0], negative['pat_id'][0]]

            # Getting the full embeddings of the patients
            pat_embeddings_list = [embeddings[pat_id] for pat_id in pat_ids]
            org_pat_embeddings_list = [original_embeddings[pat_id] for pat_id in pat_ids]
            
            # Keep a copy of the original embeddings detached from the computation graph
            #copy_anchor_emb_full = anchor_emb_full.detach().clone()

            # Extract the section-specific vectors from the embeddings
            section_embeddings = {}
            org_section_embeddings = {}
            keys = ['anchor', 'pos', 'neg']

            # Extract the section-specific vectors from the embeddings
            for key, emb, org_emb, section_id in zip(keys, pat_embeddings_list, org_pat_embeddings_list, [anchor['section_id'], positive['section_id'], negative['section_id']]):
                section_embeddings[key] = get_section_embedding(emb, section_id)
                org_section_embeddings[key] = get_section_embedding(org_emb, section_id)

            # Normalize the embeddings before calculating the loss
            for key in keys:
                section_embeddings[key] = F.normalize(section_embeddings[key].unsqueeze(0)).squeeze(0)
                org_section_embeddings[key] = F.normalize(org_section_embeddings[key].unsqueeze(0)).squeeze(0)
            
            # Forward pass
            loss = criterion(section_embeddings['anchor'],section_embeddings['pos'],section_embeddings['neg'],org_section_embeddings['anchor'],org_section_embeddings['pos'],org_section_embeddings['neg'],all_distance_pos,all_distance_neg,all_org_distance)

            # if loss.item() == 0:
            #     num_of_sections_with_zero_loss += 1

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss and other metrics to Weights and Biases
            #wandb.log({"loss": loss.item()})

            # check that the embeddings were updated
            # if torch.all(torch.eq(copy_anchor_emb_full, embeddings[anchor['pat_id'][0]])):
            #     print(f"Anchor embeddings were not updated for patient {anchor['pat_id'][0]}")
            #     raise ValueError

            #print(f'Mid Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}') # type: ignore
        #print(f"percentage of sections with zero loss: {num_of_sections_with_zero_loss / len(dataset)}")
        # Save mean of distances
        all_mean_distance_pos.append(torch.mean(torch.stack(all_distance_pos)))
        all_mean_distance_neg.append(torch.mean(torch.stack(all_distance_neg)))
        all_mean_org_distance.append(torch.mean(torch.stack(all_org_distance)))

        # print mean and std of distances epoch
        print(f"positive mean distance: {all_mean_distance_pos[-1]} std: {torch.std(torch.stack(all_distance_pos))}")
        print(f"negative mean distance: {all_mean_distance_neg[-1]} std: {torch.std(torch.stack(all_distance_neg))}")
        print(f"original mean distance: {all_mean_org_distance[-1]} std: {torch.std(torch.stack(all_org_distance))}")

        if len(all_mean_distance_pos) > 1:
            print(f"positive mean distance diff: {all_mean_distance_pos[-1] - all_mean_distance_pos[-2]}")
            print(f"negative mean distance diff: {all_mean_distance_neg[-1] - all_mean_distance_neg[-2]}")
            print(f"original mean distance diff: {all_mean_org_distance[-1] - all_mean_org_distance[-2]}")

        all_distance_pos = []
        all_distance_neg = []
        all_org_distance = []
    
    #wandb.finish()  # End Weights and Biases run
    plot_distances(all_mean_distance_pos, all_mean_distance_neg, all_mean_org_distance, output_dir)

    # Save the optimized embeddings
    print("Saving the optimized embeddings...")
    # create a tqdm progress bar
    pb = tqdm(total=len(embeddings))
    for pat_id, embedding in embeddings.items():
        os.makedirs(output_dir, exist_ok=True)
        np.save(f'{output_dir}/{pat_id}', embedding.detach().numpy())
        pb.update(1)  # update progress bar
    pb.close()  # close progress bar
    print("Done saving the optimized embeddings!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", help="path to embedding dir (patient-level matrix of sections embeddings).")
    # only patients with section embedding will be considered
    parser.add_argument("--dataset_path", help="used for sections labeling")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--margin", type=float, default=2.0, help="margin for triplet loss")
    parser.add_argument("--positive_weight", type=float, default=1.0, help="weight for the positive example of triplet loss")
    parser.add_argument("--section_are_max_seq_len", action='store_true', help="whether the sections were created with max_seq_len")
    parser.add_argument("--distance_metric", type=TripletDistanceMetric, choices=list(TripletDistanceMetric), default=TripletDistanceMetric.EUCLIDEAN, help="distance metric for triplet loss")
    parser.add_argument("--org_dist_weight", type=float, default=0.5, help="weight for the distance of the new embeddings from the original")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    section_embedding_dir = args.embedding_dir
    output_dir = section_embedding_dir.replace('sections', 'optimized_sections')
    patient_data = get_patients_dict(args.dataset_path)
    train_embedding_model(
        patient_data,
        section_embedding_dir,
        output_dir,
        epochs=args.epochs,
        lr=args.lr,
        distance_metric=args.distance_metric,
        margin=args.margin,
        positive_weight=args.positive_weight,
        org_dist_weight=args.org_dist_weight,
        section_are_max_seq_len=args.section_are_max_seq_len)