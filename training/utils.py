import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)
    for batch in loop:
        # <-- model-specific unpacking just before calling this function
        inputs, labels = batch  # or (input_ids, mask, labels), etc.
        inputs  = [x.to(device) for x in inputs]  # if tuple/list
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(*inputs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds  = torch.argmax(logits, dim=-1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_true  = []
    all_pos_probs = []  # prob for class 1

    loop = tqdm(loader, leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            if isinstance(inputs, (tuple, list)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = [inputs.to(device)]
            labels = labels.to(device)

            logits = model(*inputs)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            # Probability of positive class (index 1) for ROC AUC
            all_pos_probs.extend(probs[:, 1].cpu().numpy())

            epoch_acc = 100.0 * (torch.tensor(all_preds) == torch.tensor(all_true)).sum().item() / len(all_true)
            loop.set_postfix(loss=loss.item(), acc=epoch_acc)

    epoch_loss = running_loss / total
    epoch_acc  = accuracy_score(all_true, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_true, all_pos_probs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
