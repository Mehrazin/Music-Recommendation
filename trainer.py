import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import Music_data, create_data_loader
import torch.optim as optim
import torch.nn.functional as F
import Model
import random
from Model import create_embeding, Seq2Seq, Encoder, Decoder
import os

class Trainer:
    def __init__(self, config, model, data_iterator):
        self.config = config
        self.model = model
        self.train_iterator = data_iterator['train']
        self.test_iterator = data_iterator['test']
        self.valid_iterator = data_iterator['valid']
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
        path = os.path.join(config.exp_dir, 'loss_plot')
        self.writer = SummaryWriter(path)
        self.step = 0


    def iter(self):
        """
        iterates over one epooch
        """
        losses = []
        print_loss_total = 0
        print_loss_avg = 0
        for batch_idx, batch in enumerate(self.valid_iterator):
            users, source, target = batch
            target_len = target[0].shape[0]
            users = users.to(self.config.device)
            source = (source[0].to(self.config.device), source[1].to(self.config.device))
            target = (target[0].to(self.config.device), target[1].to(self.config.device))
            output = self.model(self.config, users,source, target)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[0][1:].reshape(-1)

            self.optimizer.zero_grad()
            loss = self.criterion(output, target)

            # Back prop
            loss.backward()
            losses.append(loss.item()/target_len)
            print_loss_total += loss.item()/target_len
            if batch_idx % self.config.print_every == 0:
                print_loss_avg = print_loss_total / self.config.print_every
                print_loss_total = 0
                print(f'    Batch: {batch_idx}      loss: {print_loss_avg}')
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # Gradient descent step
            self.optimizer.step()

            # Plot to tensorboard
            self.writer.add_scalar("Training loss", loss, global_step=self.step)
            self.step += 1
        return losses

    def save_checkpoint(self, epoch):
        checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        state = checkpoint
        print("=> Saving checkpoint")
        name = 'model_checkpoint.pth.tar'
        path = os.path.join(self.config.checkpoint_dir, str(name))
        torch.save(state, path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.reload_path)
        print("=> Loading checkpoint")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
