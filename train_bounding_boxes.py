# ======================
# IMPORTS
# ======================
# Gestion des fichiers et dossiers
import os
# PyTorch et outils pour la gestion des datasets
import torch
import torch.utils.data
# Torchvision : modèles de détection et transformations
import torchvision
# Permet de remplacer la tête de classification de Faster R-CNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Chargement et manipulation des images
from PIL import Image
# Lecture des fichiers XML (format Pascal VOC)
import xml.etree.ElementTree as ET  # Utilisé pour lire les fichiers XML


# ======================
# 1. CONFIGURATION DES CHEMINS
# ======================
# Le préfixe 'r' est indispensable sous Windows pour gérer correctement les backslashes
IMG_DIR = r"datasets\military-aircraft-recognition-dataset.train\student-dataset\images"
ANNOT_DIR = r"datasets\military-aircraft-recognition-dataset.train\student-dataset\annotations"


# ======================
# 2. CLASSE DATASET PERSONNALISÉE
# ======================
class MilitaryAircraftDataset(torch.utils.data.Dataset):
    """
    Dataset personnalisé pour la détection d'aéronefs.
    Chaque image est associée à une annotation XML décrivant
    les bounding boxes des objets présents.
    """

    def __init__(self, img_dir, annot_dir, transforms=None):
        # Chemins vers les images et annotations
        self.img_dir = img_dir
        self.annot_dir = annot_dir

        # Transformations à appliquer aux images
        self.transforms = transforms

        # Liste des fichiers images (jpg ou png), triée pour garantir un ordre stable
        all_imgs = list(sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ]))

        # Stockage de la liste finale des images
        self.imgs = all_imgs

    def __getitem__(self, idx):
        """
        Retourne un couple (image, target) pour l'indice donné.
        """

        # ======================
        # A. Chargement de l'image
        # ======================
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Ouverture de l'image et conversion en RGB
        img = Image.open(img_path).convert("RGB")

        # ======================
        # B. Chargement de l'annotation XML
        # ======================
        # Convention : image.jpg -> annotation image.xml
        annot_name = os.path.splitext(img_name)[0] + ".xml"
        annot_path = os.path.join(self.annot_dir, annot_name)

        boxes = []

        # Parsing du fichier XML (format Pascal VOC)
        try:
            tree = ET.parse(annot_path)
            root = tree.getroot()

            # Parcours de tous les objets annotés
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')

                # Extraction des coordonnées de la bounding box
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])

        except FileNotFoundError:
            # Gestion du cas où l'annotation est manquante
            print(f"Attention: Annotation non trouvée pour {img_name}")

            # Création d'un tenseur vide (aucun objet détecté)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # ======================
        # C. Conversion en tenseurs PyTorch
        # ======================
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)

        # Une seule classe : "Aircraft" (label = 1)
        # Le label 0 est réservé au fond (background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Identifiant unique de l'image
        image_id = torch.tensor([idx])

        # Aire des bounding boxes (utile pour certaines métriques COCO)
        area = (
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if num_objs > 0 else torch.tensor([0])
        )

        # Indicateur "iscrowd" (toujours 0 ici)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Dictionnaire target conforme à l'API Torchvision Detection
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # ======================
        # D. Application des transformations
        # ======================
        # Conversion PIL -> Tensor obligatoire pour Faster R-CNN
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """
        Retourne le nombre total d'images du dataset.
        """
        return len(self.imgs)


# ======================
# 3. FONCTION DE CRÉATION DU MODÈLE
# ======================
def get_model(num_classes):
    """
    Initialise un modèle Faster R-CNN pré-entraîné
    et adapte la tête de classification au nombre de classes.
    """

    # Chargement de Faster R-CNN avec backbone ResNet-50 + FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    # Nombre de features en entrée de la tête de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Remplacement de la tête de prédiction (num_classes inclut le fond)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


# ======================
# 4. FONCTION COLLATE
# ======================
def collate_fn(batch):
    """
    Fonction indispensable pour la détection d'objets.
    Elle permet de gérer des images avec un nombre variable de bounding boxes.
    """
    return tuple(zip(*batch))


# ======================
# 5. SCRIPT PRINCIPAL
# ======================
if __name__ == "__main__":

    # Sélection automatique du device (GPU si disponible)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Utilisation du device : {device}")

    # Transformation minimale : conversion PIL -> Tensor
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # Chargement du dataset personnalisé
    dataset = MilitaryAircraftDataset(
        IMG_DIR,
        ANNOT_DIR,
        transforms=data_transform
    )

    # Création du DataLoader
    # Batch size faible car la détection est coûteuse en mémoire
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,     # Compatible Windows
        collate_fn=collate_fn
    )

    # Initialisation du modèle
    num_classes = 2 # 2 classes : Aircraft + Background
    model = get_model(num_classes)
    model.to(device)

    # ======================
    # PARAMÈTRES D'APPRENTISSAGE
    # ======================
    # On ne met à jour que les paramètres entraînables
    params = [p for p in model.parameters() if p.requires_grad]

    # Optimiseur SGD (classique pour Faster R-CNN)
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Scheduler pour réduire le learning rate au fil des epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # Nombre d'époques d'entraînement
    num_epochs = 10

    # ======================
    # BOUCLE D'ENTRAÎNEMENT
    # ======================
    print("Début de l'entraînement...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        i = 0

        for images, targets in data_loader:

            # Envoi des images sur le device
            images = list(image.to(device) for image in images)

            # Envoi des annotations sur le device
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            # Calcul des pertes (classification + régression + RPN)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Rétropropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # Affichage périodique de la loss
            if i % 10 == 0:
                print(
                    f"Epoch: {epoch}, "
                    f"Iteration: {i}, "
                    f"Loss: {losses.item():.4f}"
                )
            i += 1

        # Mise à jour du learning rate
        lr_scheduler.step()

        print(
            f"--- Fin Epoch {epoch}, "
            f"Loss Moyenne: {epoch_loss/i:.4f} ---"
        )

    # ======================
    # SAUVEGARDE DU MODÈLE
    # ======================
    print("Sauvegarde du modèle...")
    torch.save(model.state_dict(), "bounding_boxess_model.pth")
    print("Terminé !")
