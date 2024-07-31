#Import_dependencies
import shutil, os, math, torch, wandb
import pandas as pd
from torch.utils.data import DataLoader
from configparser import ConfigParser

from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    )
from datasets import load_dataset

from util import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid

# enable tqdm
tqdm.pandas()

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load_config
config_object = ConfigParser()
config_object.read('config.ini')
config = config_object["CONFIG"] 

# Training params
model_name = config['model_name']
max_seq_length = int(config['max_seq_length'])
batch_sizes = [int(bs) for bs in config['batch_sizes'].split(',')]
learning_rates = [float(lr) for lr in config['learning_rates'].split(',')]
num_epochs = int(config['num_epochs'])
seeds = [int(s) for s in config['seeds'].split(',')]
warmup_ratio = float(config['warmup_ratio'])
sample_sizes = [int(ss) for ss in config['samples'].split(',')]
label_names = config['label_names']

hyperparameters = {
    'batch_size': batch_sizes,
    'learning_rate': learning_rates,
    'sample_size': sample_sizes,
    'seed': seeds,
}

# load dataset
print('Loading data...')
dataset_name = config['dataset_name']
train_split = config['train_split']
val_split = config['val_split']
test_split = config['test_split']
text_column_name = config['text_column_name']
label_column_name = config['label_column_name']
dir_out = config['dir_out']

# Prepare main output directory
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

# Prepare_dataset
data = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if not val_split.strip(): # No val split available => manually create one
    train = data[train_split].to_pandas()
    train, val = train_test_split(train, test_size=.2, random_state=42, stratify=train[label_column_name])
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    test = data[test_split].to_pandas()

else: # load val split if available
    train, val, test = data[train_split].to_pandas(), data[val_split].to_pandas(), data[test_split].to_pandas()

if dataset_name == 'empathetic_dialogues':
    data = process_dialogues(
        train=train,
        val=val,
        test=test
    )
    train, val, test = data['train'], data['val'], data['test']

data_val = ClassificationDataset(val[text_column_name], val[label_column_name], label_names, max_seq_length, tokenizer)
data_test = ClassificationDataset(test[text_column_name], test[label_column_name], label_names, max_seq_length, tokenizer)

# Retrieve num labels in order to correctly initialize model later
num_labels = len(train[label_column_name].unique())

# Fine-tuning_________________________________________________________________________________________
# Start experiment permutations
for params in ParameterGrid(hyperparameters):

    # Load hyperparameter configuration
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    sample_size = params['sample_size']
    random_seed = params['seed']

    # Set WandB logging
    wandb.init(
        project=config['wandb_project'], 
        group=f"label_{sample_size}_{learning_rate}_{batch_size}", 
        entity="jlemmens",
        name=f'{random_seed}'
        )

    # set seed
    set_seed(random_seed)

    # Sample train data
    if sample_size < len(train):
        train_sample, _ = train_test_split(train, train_size=sample_size, random_state=random_seed, stratify=train[label_column_name])
    else:
        train_sample = train.sample(frac=1, random_state=random_seed)
    train_sample = train_sample.reset_index()
    data_train = ClassificationDataset(
        train_sample[text_column_name], 
        train_sample[label_column_name], 
        label_names,
        max_seq_length, 
        tokenizer
        )

    # create dataloaders
    train_cls_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_cls_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
    test_cls_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    
    # prepare output directories
    sample_dir = f'{dir_out}/{sample_size}'
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        output_hidden_states=True
        ).to(device)
                    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Initialize linear learning rate scheduler with warmup
    num_training_steps = math.ceil(sample_size/batch_size*num_epochs) 
    num_warmup_steps = math.ceil(num_training_steps * warmup_ratio) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
        )

    # Start fine-tuning loop
    print('Fine-tuning on end task...')
    for epoch in tqdm(range(1, num_epochs+1)):

        # prepare output directories
        hyperparam_dir = os.path.join(sample_dir, f'epoch_{epoch}_bs_{batch_size}_lr_{learning_rate}/')
        if not os.path.isdir(hyperparam_dir):
            os.mkdir(hyperparam_dir)

        seed_dir = os.path.join(hyperparam_dir, f'seed_{random_seed}')
        os.mkdir(seed_dir)

        avg_ce_loss = []

        for batch_idx, batch in tqdm(enumerate(train_cls_loader)):
            
            # Training step
            ce_loss = step(
                device,
                optimizer,
                scheduler,
                model,
                batch,
                )
            
            avg_ce_loss.append(ce_loss)
            
        # Evaluate after each epoch
        print('Evaluating...')
        val_p, val_r, val_f, val_preds, val_cr = evaluate(model, val_cls_loader, device)

        # Save predictions to disk
        val_df = pd.DataFrame(data={
            'true': data_val.labels,
            'pred': val_preds,
            })

        # Log results
        avg_ce_loss = torch.stack(avg_ce_loss, dim=0)

        wandb.log({
            "val_p": val_p,
            "val_r": val_r,
            "val_f1": val_f,
            "ce_loss": torch.mean(avg_ce_loss, dim=0),
            "epoch": epoch,
            })
        
        print('Testing...')
        test_p, test_r, test_f, test_preds, test_cr = evaluate(model, test_cls_loader, device)

        # Save predictions to disk
        test_df = pd.DataFrame(data={
            'true': data_test.labels,
            'pred': test_preds,
            })

        # Log results
        wandb.log({
            "test_p": test_p,
            "test_r": test_r,
            "test_f1": test_f,
            "epoch": epoch,
            })

        val_df.to_csv(os.path.join(seed_dir, 'val_preds.csv'), index=False)
        test_df.to_csv(os.path.join(seed_dir, 'test_preds.csv'), index=False)

        val_cr.to_csv(os.path.join(seed_dir, 'val_results.csv'), index=True)
        test_cr.to_csv(os.path.join(seed_dir, 'test_results.csv'), index=True)

    wandb.finish()

# Avoid full disk
shutil.rmtree('wandb')
shutil.rmtree('__pycache__')
