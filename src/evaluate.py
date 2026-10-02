import argparse, torch
from sklearn.metrics import classification_report
from data import get_loaders
from model import build_model

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--weights", required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, test_loader, classes = get_loaders(args.data_dir)

model = build_model(len(classes)).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

preds, trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        trues += y.tolist()

print(classification_report(trues, preds, target_names=classes))
