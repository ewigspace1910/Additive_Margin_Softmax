import os
import argparse
import yaml
import tqdm
import torch
import numpy as np
from modules.evaluate import evaluate_model, ensem_evaluate_model, load_model
from modules.models import MobileFaceNet, Backbone
from modules.dataloader import get_DataLoader, ValidDataset
from torch.nn import DataParallel
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--p", type=str, help="the saved file path")
    return parser.parse_args()

def verify(cfg, nworker=2):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)
    models = []

    for model_path in cfg['weight_path']:
        if cfg['backbone'].lower() == 'resnet50':
            print("use ir-resnet50")
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], embedding_size=cfg['embd_size'], mode='ir_se')
        elif cfg['backbone'].lower() == 'resnet100':
            print("use ir-resnet100")
            backbone = Backbone(100, drop_ratio=cfg['drop_ratio'],embedding_size=cfg['embd_size'], mode='ir_se')
        else:
            print("use mobile FaceNet")
            backbone = MobileFaceNet(cfg['embd_size'])

        backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        
        backbone.eval()

        models.append(backbone)
    
    print("-Validate...") 
    with torch.no_grad():
        for x in cfg['valid_data']:
            accs, best_thresholds, eers = ensem_evaluate_model(models, valid_set[x], device=device)
            # print('\t--{}\'s max accuracy list: {:.5f} \t best threshold: {} \teer: {:.5f}'.format(x,max(accs), thrs, eer))
            print("accs: ", accs)
            print("best_thresholds: ", best_thresholds)
            print("eers: ", eers)

def main():
    pass

if __name__ == "__main__":
    args = get_args()

    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    verify(config)