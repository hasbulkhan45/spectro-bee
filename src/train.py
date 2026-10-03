import argparse, os, torch, torch.nn as nn
from tqdm import tqdm
from data import get_loaders
from model import build_model

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--out", default="best.pth")
p.add_argument("--resume", action="store_true",
               help="continue from last.pth (previous epochs)")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device, flush=True)

train_loader, val_loader, test_loader, classes = get_loaders(args.data_dir)
print(len(classes), "classes |", len(train_loader.dataset), "train images", flush=True)

model = build_model(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

last_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "last.pth")
start_epoch, best = 0, 0.0

if args.resume and os.path.exists(last_path):
    ck = torch.load(last_path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    start_epoch, best = ck["epoch"], ck["best"]
    print(f"Resumed from epoch {start_epoch}, best val acc {best:.3f}", flush=True)

def run_epoch(loader, train, desc):
    model.train(train)
    total_loss = correct = count = 0
    bar = tqdm(loader, desc=desc, leave=False, mininterval=5)
    with torch.set_grad_enabled(train):
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            count += len(y)
            bar.set_postfix(loss=f"{total_loss/count:.3f}", acc=f"{correct/count:.3f}")
    return total_loss / count, correct / count

# epochs run strictly one after another: 1, 2, 3, ...
for e in range(start_epoch, start_epoch + args.epochs):
    tl, ta = run_epoch(train_loader, True, f"epoch {e+1} train")
    vl, va = run_epoch(val_loader, False, f"epoch {e+1} val")
    print(f"epoch {e+1}: train acc {ta:.3f} | val loss {vl:.3f} val acc {va:.3f}", flush=True)

    if va > best:
        best = va
        torch.save(model.state_dict(), args.out)
        print("  saved best ->", args.out, flush=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": e + 1, "best": best}, last_path)
