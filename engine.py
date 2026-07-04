import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from utils import save_imgs
import matplotlib.pyplot as plt
import os
import math
import time  # 导入时间模块
import cv2
from skimage import morphology
import numpy as np


def clDice_metric(img, lab, beta=2):
    """计算clDice指标（修正后，结果范围 [0, 1]）"""
    # 输入二值化（假设输入为概率图，阈值 0.5）
    img = (img >= 0.5).astype(np.uint8)
    lab = (lab >= 0.5).astype(np.uint8)

    # 骨架化
    skel_pred = morphology.skeletonize(img).astype(np.uint8)
    skel_gt = morphology.skeletonize(lab).astype(np.uint8)

    # 形态学膨胀（beta 控制膨胀程度）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * beta + 1, 2 * beta + 1))
    dil_pred = cv2.dilate(skel_pred, kernel)  # 预测骨架膨胀
    dil_gt = cv2.dilate(skel_gt, kernel)  # 真实骨架膨胀

    # 计算交集：预测膨胀覆盖的真实骨架 + 真实膨胀覆盖的预测骨架
    overlap_pred = dil_pred & skel_gt  # 预测膨胀后与真实骨架的交集
    overlap_gt = dil_gt & skel_pred  # 真实膨胀后与预测骨架的交集
    intersection = np.sum(overlap_pred | overlap_gt)  # 并集去重

    # 计算并集：预测骨架 + 真实骨架的总像素数
    union = np.sum(skel_pred) + np.sum(skel_gt)

    # 避免除以零（当两者均为空时，视为完全匹配，返回 1.0）
    cldice = intersection / union if union != 0 else 1.0
    return cldice

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # 记录 epoch 开始时间
    epoch_start_time = time.time()

    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):

        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()


        gt_pre, out = model(images)
        loss = criterion(gt_pre, out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)


        # if iter % config.print_interval == 0:
        #     log_info = f'train: epoch {epoch}, iter:{iter}, ' \
        #                f'loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
        #                f'iter_time: {iter_time:.2f}s'  # 添加迭代时间
        #     print(log_info)
        #     logger.info(log_info)
        if iter % config.print_interval == 0:
            log_info = (
                f'train: epoch {epoch}, iter:{iter}, '
                f'loss: {np.mean(loss_list):.4f}, lr: {now_lr}, '

            )
            print(log_info)
            logger.info(log_info)

    # 计算整个 epoch 的耗时
    epoch_time = time.time() - epoch_start_time
    log_info = f'Epoch {epoch} finished, epoch_time: {epoch_time:.2f}s'  # 添加 epoch 时间
    print(log_info)
    logger.info(log_info)

    scheduler.step()

    # 计算该 epoch 的平均训练损失
    epoch_train_loss = np.mean(loss_list)
    return step, epoch_train_loss


    # return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        # TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
        #
        # accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        # sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        # specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        # f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        # miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        #
        # log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
        #         specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'

        # 检查混淆矩阵的形状
        if confusion.shape == (2, 2):
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                    specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        else:
            # 当混淆矩阵不是 2x2 时，给出提示信息
            print(f"Warning: Confusion matrix shape is {confusion.shape}, not able to calculate all metrics.")
            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, confusion_matrix: {confusion}'
            accuracy = 0
            sensitivity = 0
            specificity = 0
            f1_or_dsc = 0
            miou = 0

        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    # 新增clDice指标收集
    clDice_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

            # 计算clDice（逐样本计算）
            for batch_idx in range(out.shape[0]):
                single_pred = out[batch_idx]  # [H, W]
                single_gt = msk[batch_idx]    # [H, W]
                
                # 概率图转二值化（使用阈值）
                pred_binary = (single_pred >= config.threshold).astype(np.uint8)
                gt_binary = (single_gt >= 0.5).astype(np.uint8)
                
                # 计算clDice
                cldice = clDice_metric(pred_binary, gt_binary, beta=2)
                clDice_list.append(cldice)

            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
                #save_imgs(gt_pre[0], gt_pre[1], gt_pre[2], i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name+'1')
                #save_imgs(gt_pre[3], gt_pre[4], msk, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name+'2')

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        
        auc1 = roc_auc_score(y_true, preds)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0


        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        mcc = (float(TP*TN)- float(FP*FN))/math.sqrt(float(TP + FP) *float(TP + FN)*float(FP + TN)*float(FN + TN))
        # result = math.sqrt(x)
        bm = (float(TP) / float(TP + FN)) + (float(TN) / float(FP + TN)) - 1
        

        # 计算准确率和 F1 分数
        accuracy = accuracy_score(y_true, y_pre)
        f1 = f1_score(y_true, y_pre)



        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity_OR_recall: {sensitivity}, precision:{precision}, \
                confusion_matrix: {confusion},mcc: {mcc}, bm: {bm},  auc1: {auc1}'
        
                # 新增clDice结果输出
        clDice_mean = np.mean(clDice_list)
        clDice_std = np.std(clDice_list)
        
        log_info += f", clDice: {clDice_mean:.4f} ± {clDice_std:.4f}"

        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
