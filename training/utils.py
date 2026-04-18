import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
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

        # Step per-batch schedulers (e.g. cosine warmup); epoch-level ones
        # (StepLR, ReduceLROnPlateau) are stepped manually in the notebook
        if scheduler is not None and not isinstance(
            scheduler,
            (
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
        ):
            scheduler.step()

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
    all_preds = []
    all_true  = []
    all_probs = []
    total = 0

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

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_true.extend(labels.cpu().numpy().tolist())

            epoch_acc = 100.0 * sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
            loop.set_postfix(loss=loss.item(), acc=epoch_acc)

    epoch_loss = running_loss / total
    epoch_acc  = accuracy_score(all_true, all_preds)
    all_probs = np.concatenate(all_probs, axis=0)  # shape: (N, num_classes)
    return epoch_loss, epoch_acc, all_preds, all_true, all_probs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
