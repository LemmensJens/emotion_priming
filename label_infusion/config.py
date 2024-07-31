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
    'label_names': '',
    'train_split': '',
    'val_split': '',
    'test_split': '',
    'dir_out': '',
    'wandb_project': '',
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)