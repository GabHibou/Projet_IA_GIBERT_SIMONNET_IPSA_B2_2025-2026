# ======================
# IMPORTS
# ======================
# Gestion des fichiers et dossiers
import os
# PyTorch core
import torch
import torch.nn as nn
import torch.optim as optim
# Outils torchvision pour datasets, transformations et modèles pré-entraînés
from torchvision import datasets, transforms, models
# DataLoader pour charger les données par batch
from torch.utils.data import DataLoader
# Métriques de classification (précision, rappel, F1)
from sklearn.metrics import precision_recall_fscore_support


# ======================
# CONFIGURATION
# ======================
# Dossier contenant les images classées par sous-dossiers (1 dossier = 1 classe)
DATASET_DIR = "crops_aircraft"

# Taille des batchs (réduite pour un entraînement sur CPU)
BATCH_SIZE = 8

# Nombre d'époques d'entraînement
NUM_EPOCHS = 5

# Taux d'apprentissage
LEARNING_RATE = 1e-3

# Taille cible des images (carré IMG_SIZE x IMG_SIZE)
IMG_SIZE = 128

# Nom du fichier de sauvegarde du modèle entraîné
MODEL_OUTPUT = "classifier_model.pth"

# Choix du device (CPU ici)
device = torch.device("cpu")


# ======================
# TRANSFORMS
# ======================
# Transformations appliquées aux images d'entraînement
train_transforms = transforms.Compose([
    # Redimensionnement des images
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Augmentation de données : flip horizontal aléatoire
    transforms.RandomHorizontalFlip(),

    # Augmentation de données : rotation aléatoire légère
    transforms.RandomRotation(10),

    # Conversion en tenseur PyTorch
    transforms.ToTensor(),

    # Normalisation avec les statistiques ImageNet
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# Transformations appliquées aux images de validation (pas d'augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


# ======================
# DATASETS
# ======================
# Chargement du dataset complet à partir du dossier
# ImageFolder associe automatiquement chaque sous-dossier à une classe
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)

# Nombre de classes détectées
NUM_CLASSES = len(full_dataset.classes)
print("Classes détectées :", full_dataset.classes)

# Séparation train / validation (80% / 20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size]
)

# On remplace les transforms du dataset de validation
val_dataset.dataset.transform = val_transforms

# DataLoaders pour charger les données par batch
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ======================
# MODEL
# ======================
# Chargement d'EfficientNet-B1 pré-entraîné sur ImageNet
model = models.efficientnet_b1(weights="IMAGENET1K_V1")

# Récupération du nombre de features en entrée du classifieur final
in_features = model.classifier[1].in_features

# Remplacement de la dernière couche pour l'adapter au nombre de classes
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# Envoi du modèle sur le device (CPU)
model.to(device)
print("Modèle prêt ✅")


# ======================
# LOSS & OPTIMIZER
# ======================
# Fonction de perte pour la classification multi-classes
criterion = nn.CrossEntropyLoss()

# Optimiseur Adam
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ======================
# TRAINING LOOP
# ======================
# Boucle principale d'entraînement
for epoch in range(NUM_EPOCHS):

    # Mode entraînement (active dropout, batchnorm, etc.)
    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

    # Boucle sur les batchs d'entraînement
    for batch_idx, (images, labels) in enumerate(train_loader, 1):

        # Envoi des données sur le device
        images, labels = images.to(device), labels.to(device)

        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calcul de la loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        # Accumulation de la loss
        train_loss += loss.item()

        # Prédictions (classe avec le score max)
        _, predicted = outputs.max(1)

        # Calcul de l'accuracy
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Accuracy du batch
        batch_acc = 100 * predicted.eq(labels).sum().item() / labels.size(0)

        # Affichage batch par batch
        print(
            f"Batch {batch_idx}/{len(train_loader)} | "
            f"Loss: {loss.item():.4f} | "
            f"Acc: {batch_acc:.2f}%"
        )

    # ======================
    # VALIDATION
    # ======================
    # Passage en mode évaluation
    model.eval()

    val_loss = 0.0
    y_true, y_pred = [], []

    # Désactivation du calcul des gradients
    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = outputs.max(1)

            # Stockage des labels pour les métriques
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calcul des métriques macro (équilibrées entre classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Accuracy de validation
    val_acc = 100 * (
        torch.tensor(y_pred) == torch.tensor(y_true)
    ).sum().item() / len(y_true)

    # Affichage des résultats de validation
    print(
        f"Epoch {epoch+1} Validation | "
        f"Loss: {val_loss/len(val_loader):.4f} | "
        f"Acc: {val_acc:.2f}% | "
        f"P: {precision:.3f} | "
        f"R: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# ======================
# SAVE MODEL
# ======================
# Sauvegarde des poids du modèle entraîné
torch.save(model.state_dict(), MODEL_OUTPUT)
print(f"\n✅ Modèle sauvegardé : {MODEL_OUTPUT}")
