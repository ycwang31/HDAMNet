import os
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f'Epoch{epoch}', unit='batch')
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f},  lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
        pbar.set_postfix(loss=np.mean(loss_list), lr=now_lr)
        pbar.update(1)
    pbar.close()
    scheduler.step()

def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            outs = model(img)
            loss = criterion(outs, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            out = outs.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    Jaccard = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    Precision = float(TP) / float(TP + FP) if (TP + FP) != 0 else 0
    Recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    Specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    Overall_Accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    F1_score = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, Jaccard: {Jaccard}, Precision: {Precision}, Recall: {Recall}, Specificity: {Specificity}, Overall_Accuracy: {Overall_Accuracy}, F1_score= {F1_score}, confusion_matrix: {confusion}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)

def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            save_imgs(img, msk, out, i, config.work_dir + '/outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
        Jaccard = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        Precision = float(TP) / float(TP + FP) if (TP + FP) != 0 else 0
        Recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        Specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        Overall_Accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        F1_score = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0

        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, Jaccard: {Jaccard}, Precision: {Precision}, Recall: {Recall}, Specificity: {Specificity}, Overall_Accuracy: {Overall_Accuracy}, F1_score= {F1_score}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)
    return np.mean(loss_list)
