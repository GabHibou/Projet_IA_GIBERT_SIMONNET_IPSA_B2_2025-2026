import os
import xml.etree.ElementTree as ET
from PIL import Image

def resize_and_pad(image, target_size, fill_color=(0, 0, 0)):
    """
    Redimensionne une image en conservant le ratio
    puis ajoute du padding pour obtenir un carré target_size x target_size.
    Utile pour préparer des images d'entrée homogènes pour un réseau de neurones.
    """
    # Dimensions originales de l’image
    w, h = image.size

    # Facteur d’échelle basé sur le plus grand côté
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Redimensionnement avec interpolation bilinéaire
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Création d’une image carrée noire (padding)
    padded_image = Image.new("RGB", (target_size, target_size), fill_color)

    # Calcul de la position pour centrer l’image redimensionnée
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # Insertion de l’image dans le canvas
    padded_image.paste(image, (paste_x, paste_y))

    return padded_image


def create_crops(num_image, xml_path, image_path, output_dir, target_size=128):
    """
    Extrait les bounding boxes décrites dans un fichier XML (format Pascal VOC),
    crée des crops des objets détectés, les redimensionne,
    puis les sauvegarde dans des dossiers par classe.
    """
    # Chargement et parsing du fichier d’annotations XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Chargement de l’image associée
    image = Image.open(image_path).convert("RGB")

    # Parcours de chaque objet annoté dans le XML
    for i, obj in enumerate(root.findall('object')):
        # Nom de la classe (type d’avion)
        aircraft_name = obj.find('name').text

        # Création du dossier correspondant à la classe
        class_dir = os.path.join(output_dir, aircraft_name)
        os.makedirs(class_dir, exist_ok=True)

        # Récupération des coordonnées du bounding box
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Extraction de la région d’intérêt (crop)
        crop = image.crop((xmin, ymin, xmax, ymax))

        # Redimensionnement + padding pour taille uniforme
        crop = resize_and_pad(crop, target_size)

        # Nom du fichier crop (image_index_object_index)
        crop_filename = f"{num_image}_{i}.jpg"

        # Sauvegarde du crop dans le dossier de la classe
        crop.save(os.path.join(class_dir, crop_filename))

        print(f"✅ Saved: {aircraft_name}/{crop_filename}")


# -------------------- CONFIGURATION --------------------

# Dossier de sortie pour les crops organisés par classe
output_dir = 'crops_aircraft'

# Dossier contenant les fichiers XML d’annotations
annotations_dir = 'datasets/military-aircraft-recognition-dataset.train/student-dataset/annotations'

# Dossier contenant les images originales
images_dir = 'datasets/military-aircraft-recognition-dataset.train/student-dataset/images'

# Taille finale des crops (entrée du réseau)
target_size = 224


# -------------------- TRAITEMENT --------------------

# Parcours de tous les fichiers d’annotations
for xml_file in os.listdir(annotations_dir):
    # On ne traite que les fichiers XML
    if not xml_file.endswith('.xml'):
        continue

    # Identifiant de l’image (nom sans extension)
    num_image = os.path.splitext(xml_file)[0]

    # Chemins complets vers le XML et l’image correspondante
    xml_path = os.path.join(annotations_dir, xml_file)
    image_path = os.path.join(images_dir, f'{num_image}.jpg')

    # Vérification que l’image existe avant traitement
    if os.path.exists(image_path):
        create_crops(num_image, xml_path, image_path, output_dir, target_size)

print("✅ Traitement terminé.")



