
import os
from torchvision import transforms, datasets
import torch

IMAGE_ROOT_PATH = 'D:\whz\imagenet'

def load_ImageNet(ImageNet_PATH, batch_size=256, workers=3, pin_memory=True, ddp=False):
    traindir = os.path.join(ImageNet_PATH, 'ILSVRC2012_img_train')
    valdir = os.path.join(ImageNet_PATH, 'ILSVRC2012_img_val')
    print('traindir = ', traindir)
    print('valdir = ', valdir)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    )
    print('train_dataset = ', len(train_dataset))
    print('val_dataset   = ', len(val_dataset))

    if not ddp:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None
        )

    else:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=sampler
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_dataset, val_dataset

if __name__ == '__main__':
    load_ImageNet(ImageNet_PATH='D:\whz\imagenet')


