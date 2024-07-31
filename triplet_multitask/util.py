import random as rd
import numpy as np
import torch
import os
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
  def __init__(self):
    super().__init__()
    
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
         batch):

    """
    Excecute fine-tuning step.
    """

    # Forward pass
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss

    # Backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss

def evaluate(model, val_loader, device):

    """
    Evaluation step after each epoch.
    Returns precision, recall, and F1-score.
    """

    model.eval()
    
    labels = []
    preds = []
    
    with torch.no_grad():

        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probabilities = nn.functional.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            preds.extend(predictions.tolist())
            labels.extend(batch['labels'].tolist())

    precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average='macro')
    cr = classification_report(labels, preds, digits=3, output_dict=True)
    cr = pd.DataFrame(cr).transpose()

    return round(precision, 3), round(recall, 3), round(fscore, 3), preds, cr

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

def find_best_val_config_and_remove_others(out_dir, sample_sizes):

    for sample_size in sample_sizes:
        sample_dir = os.path.join(out_dir, f'sample_size_{sample_size}')
        seed_dirs = os.path.join(sample_dir, 'results_per_seed')

        # Dictionary to store average F1-scores for each epoch / batch size / learning rate combination
        avg_f1_scores = {}

        # Traverse through all the seed directories
        for seed_dir in os.listdir(seed_dirs):
            seed_path = os.path.join(seed_dirs, seed_dir)

            # Skip if the current item is not a directory
            if not os.path.isdir(seed_path):
                continue

            # Traverse through the subdirectories for each seed
            for param_dir in os.listdir(seed_path):
                param_path = os.path.join(seed_path, param_dir)

                # Skip if the current item is not a directory
                if not os.path.isdir(param_path):
                    continue

                # Load results from the parameter subdirectory (assuming results are stored in a CSV file)
                result_file = os.path.join(param_path, "val_results.csv")
                if os.path.isfile(result_file):
                    df_results = pd.read_csv(result_file)

                    # Calculate the F1-score and store it with the corresponding batch size / learning rate combination
                    f1 = df_results.at[7, "f1-score"]
                    batch_size, learning_rate, epoch = param_dir.split("_")[-5], param_dir.split("_")[-3], param_dir.split("_")[-7]
                    if (batch_size, learning_rate, epoch) in avg_f1_scores:
                        avg_f1_scores[(batch_size, learning_rate, epoch)].append(f1)
                    else:
                        avg_f1_scores[(batch_size, learning_rate, epoch)] = [f1]

        # Calculate the average F1-score for each batch size / learning rate / epoch combination
        for config, f1_scores in avg_f1_scores.items():
            avg_f1_scores[config] = mean(f1_scores)

        # Find the best configuration based on the highest F1-score averaged across seeds
        best_config = max(avg_f1_scores, key=avg_f1_scores.get)

        # Traverse through the directories again and remove the ones that do not match the best configuration
        for seed_dir in os.listdir(seed_dirs):
            seed_path = os.path.join(seed_dirs, seed_dir)

            if not os.path.isdir(seed_path):
                continue

            for param_dir in os.listdir(seed_path):
                param_path = os.path.join(seed_path, param_dir)

                if not os.path.isdir(param_path):
                    continue
                
                batch_size, learning_rate, epoch = param_dir.split("_")[-5], param_dir.split("_")[-3], param_dir.split("_")[-7]

                # Check if the current configuration matches the best configuration
                if (batch_size, learning_rate, epoch) != best_config:
                    # Remove the directory
                    os.system(f"rm -rf {param_path}")

def compute_avg_results_per_seed(dir_out, sample_sizes):

    for sample_size in sample_sizes:
        sample_dir = os.path.join(dir_out, f'sample_size_{sample_size}')
        sample_path = os.path.join(sample_dir, 'results_per_seed')

        for split in ['val', 'test']:
            p = []
            r = []
            f = []

            for seed_dir in os.listdir(sample_path):
                seed_path = os.path.join(sample_path, seed_dir)

                for config_dir in os.listdir(seed_path):
                    config_path = os.path.join(seed_path, config_dir)
                    results_df = pd.read_csv(config_path+f'/{split}_results.csv')
                    p.append(results_df.at[7, 'precision'])
                    r.append(results_df.at[7, 'recall'])
                    f.append(results_df.at[7, 'f1-score'])
            
            df = pd.DataFrame(data={
                'Pre': [round(mean(p), 3)],
                'Rec': [round(mean(r), 3)],
                'F1': [round(mean(f), 3)],
                'Std': [round(stdev(f), 3)],
            })

            dir_combined = f'{dir_out}/sample_size_{sample_size}/results_combined'
            df.to_csv(f'{dir_combined}/{split}_results.csv', index=False)

def combine_all_results(dir_out):

    for split in ['val', 'test']:
        sample_sizes, p, r, f, std = [], [], [], [], []

        for sample_dir in os.listdir(dir_out):
            sample_path = os.path.join(dir_out, sample_dir)
            if not os.path.isdir(sample_path):
                continue
            sub_df = pd.read_csv(f'{sample_path}/results_combined/{split}_results.csv')
            sample_size = int(sample_dir.split('_')[-1])
            sample_sizes.append(sample_size)
            p.append(sub_df['Pre'][0])
            r.append(sub_df['Rec'][0])
            f.append(sub_df['F1'][0])
            std.append(sub_df['Std'][0])
        
        df = pd.DataFrame(data={
            'Sample': [int(s) for s in sample_sizes],
            'Pre': p,
            'Rec': r,
            'F1': f,
            'Std': std,
        })

        df = df.sort_values(by='Sample', ascending=True)
        df.to_csv(f'{dir_out}/{split}_results_combined.csv', index=False)
        
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