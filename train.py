import os
import datetime
import torch
import torch.utils.data
import compute_mean_std
import transforms as T
from datasets import DriveDataset, Chasedb1Datasets, STAREDataset, HRFDataset
from model.FRD_Net import FRD_Net
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary

class SegmentationPresetTrain:
    # 用于图像分割任务训练阶段的数据预处理和增强操作。
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    crop_size = 400

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = FRD_Net(in_channels=3, num_classes=num_classes, base_c=64)

    summary(model, input_size=(1,3, 400, 400))  # 输入图像尺寸
    return model


def main(args):



    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size #parser.add_argument("-b", "--batch-size", default=4, type=int)
    # segmentation nun_classes + background
    num_classes = args.num_classes #parser.add_argument("--num-classes", default=1, type=int)

    model = create_model(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # using compute_mean_std.py
    # mean, std = compute_mean_std.compute()
    # mean = (0.212, 0.212, 0.212)
    # std = (0.157, 0.157, 0.157)
    # mean1 = (0.215, 0.215, 0.215)
    # std1 = (0.151, 0.151, 0.151)

    # stare
    mean = (0.795, 0.425, 0.109)
    std = (0.172, 0.113, 0.064)
    mean1 = (0.802, 0.45, 0.132)
    std1 = (0.163, 0.108, 0.046)


    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # DRIVE
    # train_dataset = DriveDataset(args.data_path, #    parser.add_argument("--data-path", default="DRIVE", help="DRIVE root")
    #                              train=True,
    #                              transforms=get_transform(train=True, mean=mean, std=std))
    #
    # val_dataset = DriveDataset(args.data_path,
    #                            train=False,
    #                            transforms=get_transform(train=False, mean=mean1, std=std1))

    # train_dataset = HRFDataset(args.data_path,  train=True,
    #                              transforms=get_transform(train=True, mean=mean, std=std)) #+
    # val_dataset = HRFDataset(args.data_path,  train=False,
    #                            transforms=get_transform(train=False, mean=mean1, std=std1))

    train_dataset = STAREDataset(args.data_path, train=True,
                               transforms=get_transform(train=True, mean=mean, std=std))  # +
    val_dataset = STAREDataset(args.data_path, train=False,
                             transforms=get_transform(train=False, mean=mean1, std=std1))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=2,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # 打印数据集的大小
    print(f'#----------Training dataset size: {len(train_dataset)}----------#')
    print(f'#----------Validation dataset size: {len(val_dataset)}----------#')

    # 打印训练集图片的大小
    print('#----------Training dataset image sizes----------#')
    for i, (images, _) in enumerate(train_loader):
        print(f"Batch {i}: {images.shape}")
        if i == 0:  # 只打印第一个批次的大小，如果需要可以取消这个条件
            break

    # 打印测试集图片的大小
    print('#----------Validation dataset image sizes----------#')
    for i, (images, _) in enumerate(val_loader):
        print(f"Batch {i}: {images.shape}")
        if i == 0:  # 只打印第一个批次的大小，如果需要可以取消这个条件
            break

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam( #这段代码使用 PyTorch 库中的 torch.optim.Adam 类创建了一个 Adam 优化器实例。
        params_to_optimize, #从模型的所有参数里筛选出需要进行梯度更新的参数，并把这些参数存储在列表 params_to_optimize 中。
        lr=args.lr
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # (Initialize logging)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_metric = {"F1": 0.5}
    loss_list = []
    # for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(args.epochs):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scheduler,
                                    scaler=scaler)
        loss_list.append(mean_loss)
        # drive
        # scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 10 == 0:
            acc, se, sp, F1, mIou, pr, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes)
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n" \
                             f"AUC: {AUC_ROC:.6f}\n" \
                             f"acc: {acc:.6f}\n" \
                             f"se: {se:.6f}\n" \
                             f"sp: {sp:.6f}\n" \
                             f"F1: {F1:.6f}\n" \
                             f"Pr: {pr:.6f}\n" \
                             f"mIou: {mIou:.6f}\n"
                f.write(train_info + "\n\n")
            print(f"AUC: {AUC_ROC:.6f}")
            print(f"acc: {acc:.6f}")
            print(f"se: {se:.6f}")
            print(f"sp: {sp:.6f}")
            print(f"mIou: {mIou:.6f}")
            print(f"F1: {F1:.6f}")

            if args.save_best is True:
                if best_metric["F1"] < F1:
                    best_metric["F1"] = F1
                else:
                    continue

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            if not os.path.exists("save_weights"):
                os.makedirs("save_weights")
            if args.save_best is True:
                torch.save(save_file, "save_weights/best_model.pth")
            else:
                torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    plt.plot(loss_list)
    # plt.show()
    plt.savefig('figure.png')  # 保存图形到文件


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch Seg-UNET training")
    parser.add_argument("--data-path", default="STARE", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", help="training device")  #
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    # parser.add_argument("--epochs", default=200, type=int, metavar="N",
    #                     help="number of total epochs to train")

    parser.add_argument("--epochs", default=250, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)',
    #                     dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--early_stop', default=35, type=int)
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use pytorch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
