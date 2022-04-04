import os
import argparse
import torch
import yaml
import tqdm
import time
from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from modules.models import Backbone
from modules.dataloader import get_DataLoader, TrainDataset, ValidDataset
from modules.metrics import CosMarginProduct, ArcMarginProduct, MagMarginProduct
from modules.evaluate import evaluate_model
from modules.focal_loss import FocalLoss

import autogluon as ag

cfg = {}
NWORKER = 2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    return parser.parse_args()
    
def main(args, reporter):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #train data
    train_dataset = TrainDataset(data_list_file=cfg['train_data'],
                       is_training=True,
                       input_shape=(3, cfg['image_size'], cfg['image_size']))
    trainloader = get_DataLoader(train_dataset,
                                   batch_size=cfg['batch_size'],
                                   shuffle=True,
                                  num_workers=NWORKER)
    #valid data
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=NWORKER)
        break

    #get backbone
    if cfg['backbone'].lower() == 'resnet50':
        print("use ir-se_Resnet50")
        backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], embedding_size=cfg['embd_size'], mode='ir_se')
    elif cfg['backbone'].lower() == 'resnet100':
        print("use ir-resnet100")
        backbone = Backbone(100, drop_ratio=cfg['drop_ratio'],embedding_size=cfg['embd_size'], mode='ir_se')
    else:
        print("backbone must resnet50, resnet100")
        exit()

    #metrics
    margin = True
    if cfg['loss'].lower() == 'cosloss':
        print("use Cos-Loss")
        partial_fc = CosMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'arcloss':
        print("use ArcLoss")
        partial_fc = ArcMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'magloss':
        print('use Mag-Loss')
        partial_fc = MagMarginProduct(in_features=cfg['embd_size'], 
                                out_features=cfg['class_num'], 
                                s=cfg['logits_scale'], 
                                l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=20)
    else:
        print("No Additative Margin")
        partial_fc = torch.nn.Linear(cfg['embd_size'], cfg['class_num'], bias=False)
        margin = False
    
    #data parapell
    backbone = DataParallel(backbone.to(device))
    partial_fc = DataParallel(partial_fc.to(device))

    #optimizer
    if 'optimizer' in cfg.keys() and cfg['optimizer'].lower() == 'adam':
        optimizer = Adam([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = SGD([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    #LossFunction+scheduerLR
    if cfg['criterion'] == 'focal':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    lr_steps = [ s for s in cfg['lr_steps']] #epochs
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    for e in range(1,cfg['epoch_num']+1):
        backbone.train()
        for data in tqdm.tqdm(iter(trainloader)):
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device).long()

            logits = backbone(inputs)
            if margin: logits = partial_fc(logits, label)
            else: logits = partial_fc(logits)

            if len(logits) == 2: 
                loss = criterion(logits[0], label) + logits[1]
                logits = logits[0]
            else: loss = criterion(logits, label) 
            #update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
        scheduler.step() 
    #test
    backbone.eval()
    acc = 0
    with torch.no_grad():
        
        for x in cfg['valid_data']:
            accs, _, _ = evaluate_model(backbone, valid_set[x], device=device)
            acc = max(accs)
            break
            
    if reporter is not None:
            # reporter enables communications with autogluon
        reporter(epoch=epoch+1, accuracy=acc)
    else:
        assert False, "hmmmm"

@ag.args(
    epochs=5,
    base_lr=ag.Real(1e-4, 1e-2, log=True),
    num_workers=NWORKER,
    optimizer=ag.Choice("SGD", "Adam"),
    drop_ratio=ag.Real(0.35, 0.6),
    logit_scale = ag.Choice(30, 63)
)    
def train_finetune(args, reporter):
    #fixed params
    cfg['epoch_num'] = args.epochs
    
    #estimator
    cfg['logits_scale'] = args.logit_scale
    cfg['optimizer']    = args.optimizer
    cfg['base_lr']      = args.base_lr
    cfg['drop_ratio']   = args.drop_ratio
    #config['batch_size']   = 128
    cfg['lr_steps']     = [9, 14]
    #run
    acc = main(args ,reporter)
    return acc
    


if __name__ == "__main__":
    args = get_args()
    with open(args.c, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.Loader)
    NWORKER = args.n
    
    myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                         resource={'num_cpus': 4, 'num_gpus': 1},
                                         num_trials=8,
                                         time_attr='epoch',
                                         reward_attr="accuracy")
    myscheduler.run()
    myscheduler.join_jobs()
    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                               myscheduler.get_best_reward()))