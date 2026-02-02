# ======================
# IMPORTS
# ======================
# Framework pour l’interface web interactive
import streamlit as st
# PyTorch et Torchvision pour les modèles de deep learning
import torch
import torchvision
# Permet de remplacer la tête de prédiction de Faster R-CNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Modèles de classification et transformations d’images
from torchvision import models, transforms
# Fonctions utilitaires (softmax)
import torch.nn.functional as F
# Manipulation des images et dessin des bounding boxes
from PIL import Image, ImageDraw
# Gestion des chemins de fichiers de manière portable
from pathlib import Path
# Utilisé pour agréger les résultats par classe
from collections import defaultdict
# Gestion du système et des signaux (fermeture propre)
import os
import signal


# -----------------------------
# CONFIGURATION GÉNÉRALE
# -----------------------------
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection avions GIBERT-SIMONNET",
    layout="wide"
)

# Seuil minimal de confiance pour conserver une détection
CROP_SCORE_THRESHOLD = 0.5

# Sélection automatique du device (GPU si disponible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Répertoire racine du projet
BASE_DIR = Path(__file__).resolve().parent

# Chemins vers les modèles entraînés
MODEL_DETECTION_PATH = BASE_DIR / "bounding_boxes_model.pth"
MODEL_CLASSIFIER_PATH = BASE_DIR / "classifier_model.pth"

# Dossiers de l’application
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR

# Dossier contenant les images à analyser
IMAGES_DIR = (
    ROOT_DIR
    / "datasets"
    / "military-aircraft-recognition-dataset.eval"
    / "eval-dataset"
    / "images"
)

# Liste des classes prédites par le classifieur
CLASS_NAMES = [
    'A1', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
    'A19', 'A2', 'A20', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'
]


# -----------------------------
# FONCTIONS UTILITAIRES
# -----------------------------
def resize_and_pad(image, target_size=128, fill_color=(0, 0, 0)):
    """
    Redimensionne une image en conservant le ratio,
    puis ajoute un padding pour obtenir une image carrée.
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    image = image.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new("RGB", (target_size, target_size), fill_color)
    padded.paste(
        image,
        ((target_size - new_w) // 2, (target_size - new_h) // 2)
    )
    return padded


def get_fasterrcnn_model(num_classes):
    """
    Initialise un modèle Faster R-CNN (ResNet50 + FPN)
    pour la détection d’aéronefs.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )
    return model


def get_classifier(num_classes):
    """
    Initialise un classifieur EfficientNet-B1
    pour identifier le type d’avion.
    """
    model = models.efficientnet_b1(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def list_images(images_dir):
    """
    Liste les images disponibles dans le dossier,
    en conservant uniquement les noms numériques.
    """
    return sorted(
        [p.stem for p in images_dir.glob("*.jpg") if p.stem.isdigit()],
        key=lambda x: int(x)
    )


# -----------------------------
# CHARGEMENT DES MODÈLES (CACHE)
# -----------------------------
@st.cache_resource
def load_models():
    """
    Charge les modèles de détection et de classification
    une seule fois grâce au cache Streamlit.
    """

    detection_model = get_fasterrcnn_model(num_classes=2)
    detection_model.load_state_dict(
        torch.load(MODEL_DETECTION_PATH, map_location=DEVICE)
    )
    detection_model.to(DEVICE).eval()

    classifier_model = get_classifier(num_classes=len(CLASS_NAMES))
    classifier_model.load_state_dict(
        torch.load(MODEL_CLASSIFIER_PATH, map_location=DEVICE)
    )
    classifier_model.to(DEVICE).eval()

    return detection_model, classifier_model


# Transformations appliquées aux crops pour la classification
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# INTERFACE STREAMLIT
# -----------------------------
st.title("Détection et classification d’avions militaires (GIBERT-SIMONNET) ✈️")

# Liste des images disponibles
image_list = list_images(IMAGES_DIR)

if not image_list:
    st.error("Aucune image trouvée dans le dossier.")
    st.stop()

# Sélection de l’image via la sidebar
num_image = st.sidebar.selectbox(
    "📂 Sélectionner une image",
    image_list
)

# Bouton de lancement de l’analyse
run_button = st.sidebar.button("Lancer l’analyse 🛫")


# -----------------------------
# EXÉCUTION DE L’ANALYSE
# -----------------------------
if run_button:
    image_path = IMAGES_DIR / f"{num_image}.jpg"

    # Chargement des modèles avec affichage d’un spinner
    with st.spinner("Chargement des modèles..."):
        detection_model, classifier_model = load_models()

    # Chargement de l’image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).to(DEVICE)

    # Détection des bounding boxes
    with torch.no_grad():
        prediction = detection_model([img_tensor])

    boxes = prediction[0]["boxes"]
    scores = prediction[0]["scores"]

    # Suppression des doublons via Non-Maximum Suppression
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.2)
    final_boxes = boxes[keep].cpu().numpy()
    final_scores = scores[keep].cpu().numpy()

    draw = ImageDraw.Draw(img)
    width, height = img.size

    results = []

    # Boucle sur chaque détection retenue
    for box, score in zip(final_boxes, final_scores):

        # Filtrage par seuil de confiance
        if score < CROP_SCORE_THRESHOLD:
            continue

        # Sécurisation des coordonnées
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Crop de l’image détectée
        crop = img.crop((x1, y1, x2, y2))
        crop = resize_and_pad(crop)

        # Préparation pour le classifieur
        crop_tensor = classifier_transform(crop).unsqueeze(0).to(DEVICE)

        # Classification du type d’avion
        with torch.no_grad():
            logits = classifier_model(crop_tensor)
            probs = F.softmax(logits, dim=1)
            class_id = probs.argmax(dim=1).item()
            class_conf = probs[0, class_id].item()

        class_name = CLASS_NAMES[class_id]
        label = f"{class_name} ({class_conf*100:.1f}%)"

        # Dessin de la bounding box et du label
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 12), label, fill="red")

        # Sauvegarde des résultats
        results.append({
            "image": crop,
            "class": class_name,
            "confidence": class_conf
        })


    # -----------------------------
    # AFFICHAGE DES RÉSULTATS
    # -----------------------------
    st.success(f"{len(results)} avion(s) détecté(s)")
    st.image(img, caption=f"Image {num_image}", use_container_width=True)

    st.subheader("🛩️ Avions potentiels détectés")

    if results:
        cols = st.columns(min(4, len(results)))
        for i, r in enumerate(results):
            with cols[i % len(cols)]:
                st.image(
                    r["image"],
                    caption=f"{r['class']} — {r['confidence']*100:.1f} %",
                    use_container_width=True
                )
    else:
        st.info("Aucun avion détecté.")

    # Résumé textuel
    st.subheader("📋 Résumé des détections")
    for i, r in enumerate(results, 1):
        st.markdown(
            f"- **Avion {i}** : `{r['class']}` "
            f"Confiance de **{r['confidence']*100:.1f} %**"
        )

    # Agrégation par classe
    summary = defaultdict(list)
    for r in results:
        summary[r["class"]].append(r["confidence"])

    st.subheader("📊 Synthèse par type d’avion")
    for cls, confs in summary.items():
        avg_conf = sum(confs) / len(confs)
        st.markdown(
            f"- **{cls}** : {len(confs)} avion(s), "
            f"Confiance moyenne **{avg_conf*100:.1f} %**"
        )


# -----------------------------
# BOUTON DE FERMETURE
# -----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("❌ Quitter l’application"):
    st.warning("Fermeture de l'application...")
    os.kill(os.getpid(), signal.SIGTERM)

