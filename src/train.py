from tqdm import tqdm
import torch.distributed as dist
from src.dataset import SimpsonsDataset
from src.simpson_net import SimpsonNet
from torch.utils.data import DataLoader
from torch import nn
import pickle

import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import multiclass_f1_score

def fit_epoch(gpu, model, train_loader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in train_loader:
        inputs = inputs.cuda(gpu)
        labels = labels.cuda(gpu)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)

    train_loss = running_loss / processed_size
    train_acc = running_corrects.double() / processed_size
    val_f1 = multiclass_f1_score(labels.data, preds)
    return train_loss, train_acc, val_f1

def eval_epoch(gpu, model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.cuda(gpu)
        labels = labels.cuda(gpu)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    val_f1 = multiclass_f1_score(labels.data, preds)
    
    return val_loss, val_acc, val_f1

def train(gpu, train_val_files, gpu_count, batch_size, epochs, label_encoder, state_path, history_path):
    rank = gpu
    world_size = gpu_count

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

    val_dataset = SimpsonsDataset(val_files, label_encoder, mode='val')
    train_dataset = SimpsonsDataset(train_files, label_encoder, mode='train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=gpu_count,
        rank=rank,
        shuffle=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=gpu_count,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, #нельзя использовать вместе с sampler
        num_workers=torch.cpu.device_count(),
        pin_memory=True,
        sampler=train_sampler,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=torch.cpu.device_count(),
        pin_memory=True,
        sampler=val_sampler
    )

    history = {
        "acc": {"train": [], "val": []},
        "loss": {"train": [], "val": []},
        "f1": {"train": [], "val": []}
    }
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} train_f1 {t_f1:0.4f} val_f1 {v_f1:0.4f}"

    torch.cuda.set_device(gpu)
    model = SimpsonNet()
    model.cuda(gpu)

    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        
            train_loss, train_acc, train_f1 = fit_epoch(gpu, model, train_loader, criterion, opt)

            val_loss, val_acc, val_f1 = eval_epoch(gpu, model, val_loader, criterion)
            
            if gpu == 0:
                history["acc"]["train"].append(train_acc.cpu())
                history["acc"]["val"].append(val_acc.cpu())
                history["loss"]["train"].append(train_loss)
                history["loss"]["val"].append(val_loss)
                history["f1"]["train"].append(train_f1.cpu())
                history["f1"]["val"].append(val_f1.cpu())
                
                pbar_outer.update(1)
                tqdm.write(log_template.format(
                    ep=epoch+1, 
                    t_loss=train_loss, 
                    v_loss=val_loss, 
                    t_acc=train_acc, 
                    v_acc=val_acc,
                    t_f1=train_f1,
                    v_f1=val_f1
                ))

    if gpu == 0:
        with open(history_path, 'wb') as file:
            pickle.dump(history, file)
        torch.save(model.module.state_dict(), state_path)