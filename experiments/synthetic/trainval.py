import argparse
import time
import os.path as osp

import torch.optim as optim
import torch as nn
import torch
import torch.nn.functional as F


from changenetwork.engine.epoch_based_trainer import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model

import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        else:
            focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class Evaluator(torch.nn.Module):
  def __init__(self):
    super(Evaluator,self).__init__()
    self.classes = ['added' , 'removed', 'nochange', 'change' , 'color_change']
    self.class_counts = {'added':0 , 'removed':0, 'nochange':0, 'change':0 , 'color_change':0}
  
  def forward(self, data_dict, output_dict):
     output_dict['output'] = F.softmax(output_dict['output'], dim = -1)
     max_index =  torch.max(output_dict['output'],dim = -1)[1]
     result_dict = {}
     if data_dict['label'] == max_index:
      name = "correct_" + self.classes[int(max_index.item())]
      print(name[8:])
      result_dict[name] = 1
      result_dict["correct"] = 1
     else:
      name = "correct_" + self.classes[int(data_dict['label'].item())]
      print(name[8:])
      result_dict[name] = 0
      result_dict["correct"] = 0
    
     result_dict[self.classes[max_index]] = 1
     return result_dict



class Trainer(EpochBasedTrainer):
  def __init__(self, cfg):
    super().__init__(cfg, max_epoch = cfg.optim.max_epoch)
    
    # dataloader
    start_time = time.time()
    train_loader, val_loader, neighbor_limits, class_weights = train_valid_data_loader(cfg)
    
    

    loading_time = time.time() - start_time
    message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
    self.logger.info(message)
    message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
    self.logger.info(message)

    self.register_loader(train_loader, val_loader)

    # model, optimizer, scheduler
    model = create_model(cfg).cuda()
    model = self.register_model(model)

    optimizer = optim.Adam(model.parameters(), lr = cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    self.register_optimizer(optimizer)

    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
    self.register_scheduler(scheduler)

    # loss function, evaluator
    #self.loss_func = torch.nn.CrossEntropyLoss().cuda()
    self.loss_func = FocalLoss()
    eval = Evaluator()
    self.evaluator = eval.cuda()
    

  def train_step(self,data_dict):
    output_dict = self.model(data_dict)
    
    label = data_dict['label'].long()
    output = output_dict['output'].view(1,-1)
    loss = self.loss_func(output, label)
    loss_dict  = {'loss' : loss}
    
    return output_dict, loss_dict
  
  def val_step(self, data_dict):
    output_dict = self.model(data_dict)

    label = data_dict['label'].long()
    output = output_dict['output'].view(1,-1)
    loss = self.loss_func(output, label)
    loss_dict  = {'loss' : loss}
    result_dict = self.evaluator(data_dict, output_dict)

    loss_dict.update(result_dict)

    return output_dict, loss_dict

  

def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()

