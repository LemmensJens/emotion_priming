from configparser import ConfigParser

config_object = ConfigParser()

config_object["CONFIG"] = {
    'model_name': '',
    'max_seq_length': '',
    'batch_size': '',
    'learning_rate': '',
    'triplet_margin': '',
    'num_epochs': '',
    'seeds': '',
    'samples': '',
    'dataset_name': '',
    'text_column_name': '',
    'label_column_name': '', 
    'train_split': '',
    'val_split': '',
    'dir_out': '',
    'wandb_project': '',
}

with open('triplet_config.ini', 'w') as conf:
    config_object.write(conf)
