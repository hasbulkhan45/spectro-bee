import argparse, torch, torch.nn as nn
from data import get_loaders
from model import build_model

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--out", default="best.pth")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device, flush=True)
train_loader, val_loader, test_loader, classes = get_loaders(args.data_dir)
print(len(classes), "classes", flush=True)

model = build_model(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def run_epoch(loader, train):
    model.train(train)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count

best = 0
for e in range(args.epochs):
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    print(f"epoch {e+1}: train acc {ta:.3f} | val loss {vl:.3f} val acc {va:.3f}", flush=True)
    if va > best:
        best = va
        torch.save(model.state_dict(), args.out)
        print("saved", args.out, flush=True)
