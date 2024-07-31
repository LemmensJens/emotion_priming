import random as rd
import numpy as np
import torch
import os, shutil, itertools
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report
from statistics import mean, stdev

eps = 1e-8 # an arbitrary small value to be used for numerical stability tricks

def cosine_distance_matrix(x):
    """Compute the cosine distance matrix
    Args:
        x: Input tensor of shape (batch_size, embedding_dim)
    
    Returns:
        Distance matrix of shape (batch_size, batch_size)
    """
    # Compute the cosine similarity matrix
    cosine_similarity = torch.nn.functional.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)

    # Compute the cosine distance matrix
    cosine_distance = 1 - cosine_similarity

    return cosine_distance

def get_triplet_mask(labels):
  """compute a mask for valid triplets
  Args:
    labels: Batch of integer labels. shape: (batch_size,)
  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices

  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  # step 2 - get a mask for valid anchor-positive-negative triplets

  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask

class BatchAllTtripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss
  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin):
    super().__init__()
    self.margin = margin
    
  def forward(self, embeddings, labels):
    """computes loss value.
    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
    Returns:
      Scalar loss value.
    """
    # step 1 - get distance matrix
    # shape: (batch_size, batch_size)
    distance_matrix = cosine_distance_matrix(embeddings)

    # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

    # shape: (batch_size, batch_size, 1)
    anchor_positive_dists = distance_matrix.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    anchor_negative_dists = distance_matrix.unsqueeze(1)
    # get loss values for all possible n^3 triplets
    # shape: (batch_size, batch_size, batch_size)
    triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

    # step 3 - filter out invalid or easy triplets by setting their loss values to 0

    # shape: (batch_size, batch_size, batch_size)
    mask = get_triplet_mask(labels)
    triplet_loss *= mask
    # easy triplets have negative loss values
    triplet_loss = nn.functional.relu(triplet_loss)

    # step 4 - compute scalar loss value by averaging positive losses
    num_positive_losses = (triplet_loss > eps).float().sum()
    triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

    return triplet_loss

def set_seed(seed):

    """
    Manually set seed to ensure reproducibility.
    """

    rd.seed(seed)    
    np.random.seed(seed) 

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def step(device, 
         optimizer,
         scheduler, 
         model,
         batch,
         margin):

    """
    Excecute fine-tuning step.
    """

    # Forward pass
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
    triplet_loss=BatchAllTtripletLoss(margin=margin)
    triplet_loss = triplet_loss(outputs.hidden_states[-1][:, 0, :], batch['labels'])

    # Backward pass
    triplet_loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return triplet_loss

def process_dialogues(**kwargs):
    
    """
    Preprocesses the EmpatheticDialogs dataset. 
    Accepts train, val, test splits as kwargs.
    Removes irrelevant columns, 
    converts string labels to one-hot-encodings, 
    removes "_comma_" from utterances.
    Output is a dictionary of split names as keys
    and dataframes as values.
    """
    
    output = {}
    label_to_id = {}
    
    for split, df in kwargs.items():
        
        if label_to_id == {}:
            labels = df.context.tolist()
            idx = 0
            
            for l in labels:
                if l not in label_to_id.keys():
                    label_to_id[l] = idx
                    idx += 1
        
        df = df[['context', 'prompt']]
        df.rename(columns={'context': 'label', 'prompt': 'text'}, inplace=True)
        df.drop_duplicates(inplace=True, keep='first')
        df.loc[:, 'text']  = df['text'].apply(lambda x: x.replace('_comma_', ''))
        df.loc[:, 'label'] = df['label'].apply(lambda x: label_to_id[x])
        output[split] = df.reset_index()
    
    return output
        
# Prepare_data__________________________________________________________________________   
class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, p_dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.linear(x)
        return logits
    
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, max_length, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode the input
        encoded_input = self.tokenizer.encode_plus(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoded_input["input_ids"].squeeze(),
            'attention_mask': encoded_input["attention_mask"].squeeze(),
            'labels': torch.tensor(label)
        }