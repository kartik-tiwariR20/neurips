"""
evaluate.py
-----------
Two downstream evaluation regimes, both important per the study design:

  linear_probe : freeze the encoder, train only a linear classifier on top
                 of its representation. Tests raw representation quality.
  fine_tune    : unfreeze the encoder too, train end-to-end on the labeled
                 split. Tests whether collapse still matters once the
                 encoder itself can adapt.

Both report accuracy on the held-out probe_test split.
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _run_classifier_training(encoder, num_classes, train_loader, val_loader,
                              test_loader, device, epochs=30, lr=1e-3,
                              train_encoder=False):
    encoder = copy.deepcopy(encoder).to(device)
    encoder.train(train_encoder)
    for p in encoder.parameters():
        p.requires_grad = train_encoder

    head = nn.Linear(encoder.out_dim, num_classes).to(device)
    params = list(head.parameters()) + (list(encoder.parameters()) if train_encoder else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1, None
    for epoch in range(epochs):
        encoder.train(train_encoder)
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if train_encoder:
                h = encoder(x)
            else:
                with torch.no_grad():
                    h = encoder(x)
            logits = head(h)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = _accuracy(encoder, head, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    test_acc = _accuracy(encoder, head, test_loader, device)
    return test_acc, best_val_acc


@torch.no_grad()
def _accuracy(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h = encoder(x)
        preds = head(h).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / total


def evaluate_representation(encoder, num_classes, train_ds, val_ds, test_ds,
                             device, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    linear_test_acc, linear_val_acc = _run_classifier_training(
        encoder, num_classes, train_loader, val_loader, test_loader,
        device, train_encoder=False,
    )
    finetune_test_acc, finetune_val_acc = _run_classifier_training(
        encoder, num_classes, train_loader, val_loader, test_loader,
        device, train_encoder=True,
    )
    return {
        "linear_probe_test_acc": linear_test_acc,
        "linear_probe_val_acc": linear_val_acc,
        "finetune_test_acc": finetune_test_acc,
        "finetune_val_acc": finetune_val_acc,
    }
