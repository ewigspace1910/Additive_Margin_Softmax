import os
import torch
import numpy as np
import time
import tqdm

def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

def calculate_accuracy(threshold, dists, actual_issame):
    predict_issame = np.less(dists, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(predict_issame),np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    
    acc = float(tp + tn) / dists.size
    return tpr, fpr, acc


def calculate_roc(thresholds, dists, actual_issame):
    tprs = np.zeros(thresholds.shape)
    fprs = np.zeros(thresholds.shape)
    accuracy = np.zeros(thresholds.shape)

    for i, thres in enumerate(thresholds):
        tprs[i], fprs[i], accuracy[i] = calculate_accuracy(thres, dists, actual_issame)
    
    best_threshold = thresholds[np.argmax(accuracy)]
    #tpr = np.mean(tprs)
    #fpr = np.mean(fprs)
    return tprs, fprs, accuracy, best_threshold


def calculate_eer(tprs, fprs):
    """FNR = FPR"""
    fnrs = 1. - tprs
    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fprs[np.nanargmin(np.absolute((fnrs - fprs)))]
    eer_2 = fnrs[np.nanargmin(np.absolute((fnrs - fprs)))]
    return (eer_1 + eer_2) / 2


def evaluate_model(model, dataset, device=torch.device('cpu')):
    '''return acc_array, best_threshold, err_value '''
    dists = np.array([]) #distants
    labels = np.array([]) #labels
    
    for img1, img2, label in tqdm.tqdm(dataset):
        label = label.cpu().data.numpy() == 1
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        embds_1 = model(img1)
        embds_2 = model(img2)

        embds_1 = embds_1.cpu().data.numpy()
        embds_2 = embds_2.cpu().data.numpy()
        
        embds_1 = l2_norm(embds_1)
        embds_2 = l2_norm(embds_2)

        diff = np.subtract(embds_1, embds_2)
        dist = np.sum(np.square(diff), axis=1)

        labels = np.hstack((labels, label))
        dists  = np.hstack((dists, dist))

    
    thresholds = np.arange(0, 4, 0.01)
    tprs, fprs, accs, best_threshold = calculate_roc(thresholds, dists, labels)
    eer = calculate_eer(tprs, fprs)
    return accs, best_threshold, eer


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)