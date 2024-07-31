from configparser import ConfigParser

config_object = ConfigParser()

config_object["CONFIG"] = {
    'model_name': '',
    'max_seq_length': '',
    'batch_sizes': '',
    'learning_rates': '',
    'num_epochs': '',
    'warmup_ratio': '',
    'seeds': '',
    'samples': '',
    'dataset_name': '',
    'text_column_name': '',
    'label_column_name': '', 
    'train_split': '',
    'val_split': '',
    'test_split': '',
    'dir_out': '',
    'wandb_project': '',
    'triplet_margin': '',
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)
