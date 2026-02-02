import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
import os
import json

# -----------------------------
# CONFIGURATION
# -----------------------------

# Seuil minimal de confiance pour conserver un bounding box détecté
CROP_SCORE_THRESHOLD = 0.5

# Choix automatique du device (GPU si disponible, sinon CPU)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dossier contenant les images à évaluer
IMAGE_DIR = r"datasets\military-aircraft-recognition-dataset.eval\eval-dataset\images"

# Dossier de sortie pour les crops d’objets détectés
OUTPUT_CROP_DIR = "crops"

# Chemin du fichier JSON final contenant les annotations
JSON_OUTPUT_PATH = "result_evaluation.json"

# Chemins des modèles entraînés
MODEL_DETECTION_PATH = "bounding_boxes_model.pth"
MODEL_CLASSIFIER_PATH = "classifier_model.pth"

# Liste des classes possibles pour la classification
CLASS_NAMES = [
    'A1','A10','A11','A12','A13','A14','A15','A16','A17','A18',
    'A19','A2','A20','A3','A4','A5','A6','A7','A8','A9'
]

# Création du dossier de sortie pour les crops s’il n’existe pas
os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

# -----------------------------
# FONCTIONS UTILITAIRES
# -----------------------------

def resize_and_pad(image, target_size=128, fill_color=(0, 0, 0)):
    """
    Redimensionne une image en conservant le ratio
    puis ajoute un padding pour obtenir une image carrée (target_size x target_size)
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    padded.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded

def get_fasterrcnn_model(num_classes):
    """
    Crée un modèle Faster R-CNN avec ResNet50-FPN
    et remplace la tête de classification pour s’adapter au nombre de classes
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_classifier(num_classes):
    """
    Crée un classifieur EfficientNet-B1
    avec une tête de classification adaptée au nombre de classes
    """
    model = models.efficientnet_b1(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model

# -----------------------------
# CHARGEMENT DES MODÈLES
# -----------------------------

# Modèle de détection (bounding boxes)
detection_model = get_fasterrcnn_model(num_classes=2)
detection_model.load_state_dict(torch.load(MODEL_DETECTION_PATH, map_location=DEVICE))
detection_model.to(DEVICE).eval()

# Modèle de classification des objets détectés
classifier_model = get_classifier(num_classes=len(CLASS_NAMES))
classifier_model.load_state_dict(torch.load(MODEL_CLASSIFIER_PATH, map_location=DEVICE))
classifier_model.to(DEVICE).eval()

# Transformations appliquées aux crops avant classification
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Moyennes ImageNet
        std=[0.229, 0.224, 0.225]   # Écarts-types ImageNet
    )
])

# -----------------------------
# TRAITEMENT DES IMAGES
# -----------------------------

# Liste triée des images à traiter
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

# Dictionnaire contenant toutes les annotations finales
annotations = {}

for image_name in image_files:
    print(f"Traitement : {image_name}")
    image_path = os.path.join(IMAGE_DIR, image_name)

    # Chargement de l’image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur chargement {image_name} : {e}")
        annotations[image_name] = []
        continue

    width, height = img.size

    # Conversion de l’image en tenseur PyTorch
    img_tensor = transforms.ToTensor()(img).to(DEVICE)

    # Inférence du modèle de détection
    with torch.no_grad():
        prediction = detection_model([img_tensor])

    # Récupération des bounding boxes et scores
    boxes = prediction[0]["boxes"]
    scores = prediction[0]["scores"]

    # Suppression des doublons via Non-Maximum Suppression (NMS)
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.2)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    annotations[image_name] = []

    # Dossier de sortie spécifique à l’image
    image_crop_dir = os.path.join(OUTPUT_CROP_DIR, image_name.replace(".jpg", ""))
    os.makedirs(image_crop_dir, exist_ok=True)

    saved = 0

    # Parcours de chaque bounding box conservée
    for box, score in zip(boxes, scores):
        if score < CROP_SCORE_THRESHOLD:
            continue

        # Coordonnées du bounding box
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Extraction et prétraitement du crop
        crop = img.crop((x1, y1, x2, y2))
        crop = resize_and_pad(crop)

        crop_tensor = classifier_transform(crop).unsqueeze(0).to(DEVICE)

        # Classification du crop
        with torch.no_grad():
            logits = classifier_model(crop_tensor)
            probs = F.softmax(logits, dim=1)
            class_id = probs.argmax(dim=1).item()
            class_conf = probs[0, class_id].item()

        class_name = CLASS_NAMES[class_id]

        # Sauvegarde des annotations
        annotations[image_name].append({
            "class": class_name,
            "coordinates": {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2
            }
        })

        # Sauvegarde du crop avec nom informatif
        crop_filename = f"{class_name}_{saved+1:02d}_{class_conf:.2f}.jpg"
        crop.save(os.path.join(image_crop_dir, crop_filename))
        saved += 1

# -----------------------------
# SAUVEGARDE JSON
# -----------------------------

# Écriture du fichier JSON final contenant toutes les prédictions
with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print(f"\nJSON final sauvegardé : {JSON_OUTPUT_PATH}")

