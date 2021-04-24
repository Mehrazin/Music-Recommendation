from utils import Config, bool_flag
import numpy as np
from dataloader import Vocab
import pickle
import os
import time
import argparse
import torch
from Model import build_model
import random
from dataloader import create_data_loader
from trainer import Trainer

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=bool_flag, default=True)
    parser.add_argument('--load_vocab', type=bool_flag, default=True)
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--load_model', type=bool_flag, default=False)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=50)
    return parser.parse_args()

def set_seed(config):
    """Set seed"""
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f'set seed to {config.seed}')

def load_data_iterator(config):
    tasks = ['train', 'test', 'valid']
    data_iterator = {}
    for task in tasks:
        data_iterator[task] = create_data_loader(config, dtype = task, shuffle = True)
    print('Data iterators have been created')
    return data_iterator


def main(config):
    data_iterator = load_data_iterator(config)
    model = build_model(config)
    trainer = Trainer(config, model, data_iterator)
    # evaluator = Eval()
    if config.train :
        assert config.eval_only == False
        print('Training Started')
        if config.load_model:
            trainer.load_checkpoint()
        for epoch in range(config.num_epochs):
            losses = trainer.iter()
            if config.save_periodic:
                if epoch%config.save_every == 0:
                    trainer.save_checkpoint(epoch)
                    print(f'Model saved for epoch = {epoch}')


    elif config.eval_only:
        # evaluator
        print('Evaluation')







if __name__ == '__main__':
    arg = arg_parse()
    config = Config(arg)
    set_seed(config)
    main(config)
