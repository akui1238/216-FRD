mean = (0.212, 0.212, 0.212)
std = (0.157, 0.157, 0.157)
mean1 = (0.215, 0.215, 0.215)
std1 = (0.151, 0.151, 0.151)

# DRIVE
train_dataset = DriveDataset(args.data_path,
                             train=True,
                             transforms=get_transform(train=True, mean=mean, std=std))

val_dataset = DriveDataset(args.data_path,
                           train=False,
                           transforms=get_transform(train=False, mean=mean1, std=std1))

def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)

class SegmentationPresetTrain:
    # 用于图像分割任务训练阶段的数据预处理和增强操作。
    def __init__(self, crop_size,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []

        trans.extend([
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

class Compose(object):
    # 将多个图像变换操作组合在一起，按顺序依次对图像和目标进行处理。
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target,  = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    # 对图像进行归一化处理，同时将目标的像素值进行简单的缩放和类型转换。
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = (target / 255).long()
        return image, target