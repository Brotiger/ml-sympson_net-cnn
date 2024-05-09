from tqdm import tqdm
from torch.distributed as dist
from src.dataset import SimpsonsDataset
from src.simpson_net import SimpsonNet
from torch.utils.data import DataLoader
from torch.nn import nn

import torch

def fit_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc

def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc

def train(gpu, train_files, val_files, gpu_count, batch_size, epochs):
    rank = gpu
    word_size = gpu_count

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        word_size=word_size,
        rank=rank
    )
    
    val_dataset = SimpsonsDataset(val_files, mode='val')
    train_dataset = SimpsonsDataset(train_files, mode='train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=gpu_count,
        rank=rank
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=gpu_count,
        rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=val_sampler
    )

    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    torch.cuda.set_device(gpu)
    model = SimpsonNet()
    model.cuda(gpu)

    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
            if gpu == 0:
                print("loss", train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, train_acc, val_loss, val_acc))
            
            if gpu == 0:
                pbar_outer.update(1)
                tqdm.write(log_template.format(
                    ep=epoch+1, 
                    t_loss=train_loss, 
                    v_loss=val_loss, 
                    t_acc=train_acc, 
                    v_acc=val_acc,
                ))

    return history