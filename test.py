import os
import argparse
import yaml
import tqdm
import torch
import numpy as np
from modules.evaluate import evaluate_model, load_model
from modules.models import MobileFaceNet, Backbone
from modules.dataloader import get_DataLoader, ValidDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    return parser.parse_args()
    
def verify(cfg, path, nworker=2):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)
                                
    if cfg['backbone'].lower() == 'resnet50':
        print("use ir-resnet50")
        backbone = Backbone(50, 0.5, embedding_size=cfg['embd_size'], mode='ir_se')
    elif cfg['backbone'].lower() == 'resnet100':
        print("use ir-resnet100")
        backbone = Backbone(100, 0.5,embedding_size=cfg['embd_size'], mode='ir_se')
    else:
        print("use mobile FaceNet")
        backbone = MobileFaceNet(cfg['embd_size'])
    
    backbone = backbone.to(device)
    load_model(backbone, path)
    
    backbone.eval()
    print("-Validate...") 
    with torch.no_grad():
        for x in cfg['valid_data']:
            acc, accs, thrs = evaluate_model(backbone, valid_set[x], device=device)
            print('\t--{}\'s max accuracy: {:.5f} \t best threshold: {} \tavg_acc: {:.5f}'.format(x,max(accs), thrs, acc))

def main():
    pass

if __name__ == "__main__":
    args = get_args()
    path = "./save/model_ir_se50.pth"
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    verify(config, path, nworker=args.n)