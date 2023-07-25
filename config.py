import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'log': (None, 'None for no logging'),
        'lr': (0.001, 'learning rate'),
        'batch-size': (10000, 'batch size'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0.005, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
        'device': (0, 'which device'),
    },
    'model_config': {
        'model': ('LogiRec', 'model name'),
        'embedding_dim': (64, 'user item embedding dimension'),
        'scale': (0.1, 'scale for init'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
        'num-layers': (3,  'number of hidden layers in encoder'),
        'margin': (0.1, 'margin value in the metric learning loss'),
        'optim': ('rsgd', 'optimizer choice'),
        'lam': (0.1, 'lam for logical relation modeling loss'),
        'digit': (1.3, 'consistency number'),
    },
    'data_config': {
        'dataset': ('ciao', 'which dataset to use'),
        'num_neg': (1, 'number of negative samples'),
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
