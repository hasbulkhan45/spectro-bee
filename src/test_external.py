import argparse, os, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from model import build_model

MAP = {
    "Apple_Scab_Leaf": "Apple___Apple_scab",
    "Apple_leaf": "Apple___healthy",
    "Apple_rust_leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper_leaf": "Pepper,_bell___healthy",
    "Bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry_leaf": "Blueberry___healthy",
    "Cherry_leaf": "Cherry_(including_sour)___healthy",
    "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
    "Peach_leaf": "Peach___healthy",
    "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight",
    "Raspberry_leaf": "Raspberry___healthy",
    "Soyabean_leaf": "Soybean___healthy",
    "Squash_Powdery_mildew_leaf": "Squash___Powdery_mildew",
    "Strawberry_leaf": "Strawberry___healthy",
    "Tomato_Early_blight_leaf": "Tomato___Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_leaf": "Tomato___healthy",
    "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "grape_leaf": "Grape___healthy",
    "grape_leaf_black_rot": "Grape___Black_rot",
}

p = argparse.ArgumentParser()
p.add_argument("--plantdoc_dir", nargs="+", required=True)
p.add_argument("--pv_dir", required=True)
p.add_argument("--weights", required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = sorted(d for d in os.listdir(args.pv_dir) if os.path.isdir(os.path.join(args.pv_dir, d)))
idx = {c: i for i, c in enumerate(classes)}
assert len(classes) == 38, f"expected 38 classes, got {len(classes)}"

def list_images(root_dir):
    out = []
    for folder in sorted(os.listdir(root_dir)):
        if folder not in MAP: print("skipping:", folder); continue
        fdir = os.path.join(root_dir, folder)
        out += [(os.path.join(fdir, f), idx[MAP[folder]]) for f in os.listdir(fdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out

samples = [s for d in args.plantdoc_dir for s in list_images(d)]
print(len(samples), "PlantDoc images")

tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class DS(Dataset):
    def __len__(self): return len(samples)
    def __getitem__(self, i): return tfm(Image.open(samples[i][0]).convert("RGB")), samples[i][1]

loader = DataLoader(DS(), batch_size=32, num_workers=2)
model = build_model(len(classes)).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

preds, trues = [], []
with torch.no_grad():
    for x, y in loader: preds += model(x.to(device)).argmax(1).cpu().tolist(); trues += y.tolist()

print(f"\nPlantDoc accuracy: {accuracy_score(trues, preds):.3f}  (PlantVillage test was 0.98)\n")
labels = sorted(set(trues) | set(preds))
print(classification_report(trues, preds, labels=labels, target_names=[classes[i] for i in labels], zero_division=0))
