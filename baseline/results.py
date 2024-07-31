import os
import pandas as pd
import numpy as np

# Define the directory where your output data is stored
output_dir = ''

# Initialize a dictionary to store the results
results_dict = {'n_samples': [], 'pre': [], 'rec': [], 'f1': [], 'std': [], 'params': []}

# Loop through the subdirectories based on sample size
sample_sizes = [folder for folder in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, folder))]

for sample_size in sample_sizes:
    sample_dir = os.path.join(output_dir, sample_size)
    
    # Initialize variables to keep track of the best hyperparameters and their validation performance
    best_hyperparams = None
    best_val_f1 = -1.0
    
    for hyperparam_folder in os.listdir(sample_dir):
        hyperparam_dir = os.path.join(sample_dir, hyperparam_folder)
        
        # Initialize lists to store F1 scores for each seed
        f1_scores = []
        
        for seed_folder in os.listdir(hyperparam_dir):
            seed_dir = os.path.join(hyperparam_dir, seed_folder)
            
            # Load the validation results
            val_results = pd.read_csv(os.path.join(seed_dir, 'val_results.csv'))
            # Extract the F1 score for 'macro avg'
            macro_f1 = val_results.loc[val_results['Unnamed: 0'] == 'macro avg', 'f1-score'].values[0]
            
            # Append the F1 score to the list
            f1_scores.append(macro_f1)
        
        # Calculate the mean F1 score across seeds for this hyperparameter configuration
        mean_f1 = np.mean(f1_scores)
        
        # Check if this hyperparameter configuration has the best validation performance
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            best_hyperparams = hyperparam_folder
        
    # Now that we have the best hyperparameter configuration, calculate the test set metrics
    best_hyperparam_dir = os.path.join(sample_dir, best_hyperparams)
    
    # Initialize lists to store test set metrics for each seed
    test_precisions = []
    test_recalls = []
    test_f1s = []
    
    for seed_folder in os.listdir(best_hyperparam_dir):
        seed_dir = os.path.join(best_hyperparam_dir, seed_folder)
        
        # Load the test results
        test_results = pd.read_csv(os.path.join(seed_dir, 'test_results.csv'))
        
        # Extract precision, recall, and F1 score for 'macro avg'
        macro_precision = test_results.loc[test_results['Unnamed: 0'] == 'macro avg', 'precision'].values[0]
        macro_recall = test_results.loc[test_results['Unnamed: 0'] == 'macro avg', 'recall'].values[0]
        macro_f1 = test_results.loc[test_results['Unnamed: 0'] == 'macro avg', 'f1-score'].values[0]
        
        # Append the metrics to the respective lists
        test_precisions.append(macro_precision)
        test_recalls.append(macro_recall)
        test_f1s.append(macro_f1)
    
    # Calculate the mean and standard deviation of test set metrics across seeds
    mean_test_precision = np.mean(test_precisions)
    mean_test_recall = np.mean(test_recalls)
    mean_test_f1 = np.mean(test_f1s)
    std_test_f1 = np.std(test_f1s)
    
    # Append the results to the dictionary
    results_dict['n_samples'].append(int(sample_size))
    results_dict['pre'].append(round((mean_test_precision*100), 1))
    results_dict['rec'].append(round((mean_test_recall*100), 1))
    results_dict['f1'].append(round((mean_test_f1*100), 1))
    results_dict['std'].append(round((std_test_f1*100), 1))
    results_dict['params'].append(best_hyperparams)

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results_dict)
results_df = results_df.sort_values(by='n_samples')

# Save the results to a CSV file
save_dir = os.path.join(output_dir, 'results.csv')
results_df.to_csv(save_dir, index=False)
