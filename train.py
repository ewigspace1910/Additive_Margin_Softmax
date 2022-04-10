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
#from modules.utils import set_memory_growth

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#set_memory_growth()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    return parser.parse_args()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.module.state_dict(), save_name)
    return save_name

def main(cfg, n_workers=2):
    #setup device
    device = torch.device("cuda") #if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    #setup path
    save_path = os.path.join(os.getcwd(), "save", cfg['model_name'])
    ckpt_path = os.path.join(save_path, "ckpt")
    log_path  = os.path.join(save_path, "log")
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    #train data
    train_dataset = TrainDataset(data_list_file=cfg['train_data'],
                       is_training=True,
                       input_shape=(3, cfg['image_size'], cfg['image_size']))
    trainloader = get_DataLoader(train_dataset,
                                   batch_size=cfg['batch_size'],
                                   shuffle=True,
                                  num_workers=n_workers)
    #valid data
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=n_workers)

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

    print(lr_steps)
    #loop
    steps_per_epoch = cfg['sample_num'] // cfg['batch_size'] + 1
    stps = steps_per_epoch if str(cfg["step_per_save"]) == "epoch" else int(cfg['step_per_save']) 
    max_acc = 0.
    writer = SummaryWriter(log_path)
    for e in range(1,cfg['epoch_num']+1):

        print("Epoch: {}/{} \n-LR: {:.6f} \n-Train...".format(e,cfg['epoch_num'], scheduler.get_last_lr()[0]))
        backbone.train()
        total_loss = 0.0
        num_batchs = 0
        num_correct = 0.
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
            
            #update metrics
            total_loss += loss.item()
            num_batchs += 1
            indices = torch.max(logits, 1)[1]
            num_correct += torch.sum(torch.eq(indices, label).view(-1)).item()
            
            #update weights
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
            
            if num_batchs % stps == 0:    # every 1000 mini-batches...
                # ...log the running loss
                writer.add_scalar('training loss',
                            total_loss / num_batchs,
                            (e-1) * len(trainloader) + num_batchs)
                writer.add_scalar('learning rate',
                            scheduler.get_last_lr()[0],
                            (e-1) * len(trainloader) + num_batchs)
        scheduler.step() 
        save_model(backbone,  ckpt_path, cfg['model_name'], e) 
        #test
        backbone.eval()
        print("-Validate...") 
        with torch.no_grad():
            for x in cfg['valid_data']:
                acc, _ , eer = evaluate_model(backbone, valid_set[x], device=device)
                acc = max(acc)
                writer.add_scalar('verification accuracy _ {} dataset'.format(x), acc, e * num_batchs)
                writer.add_scalar('verification EER _ {} dataset'.format(x), eer, e * num_batchs)
                print('\t--{}\'s accuracy: {:.5f} \t eer ~ {:.5f} '.format(x, acc, eer))
                
        print("\t--Train Loss: {:.5f} \n\t--Train accuracy: {:.5f}".format(total_loss / num_batchs, num_correct / cfg['sample_num']))

    writer.close()


if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    print(config)
    main(config ,n_workers=args.n)