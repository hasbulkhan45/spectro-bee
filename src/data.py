import copy, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_loaders(data_dir, batch_size=64, num_workers=4):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), norm])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    full_train = datasets.ImageFolder(data_dir, transform=train_tfm)  # scans once
    full_val = copy.copy(full_train)       # shares the file list, no rescan
    full_val.transform = val_tfm

    n = len(full_train)
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=g).tolist()
    a, b = int(0.8 * n), int(0.9 * n)

    mk = lambda ds, ids, sh: DataLoader(Subset(ds, ids), batch_size=batch_size,
                                        shuffle=sh, num_workers=num_workers,
                                        pin_memory=True)
    return (mk(full_train, idx[:a], True),
            mk(full_val, idx[a:b], False),
            mk(full_val, idx[b:], False),
            full_train.classes)
