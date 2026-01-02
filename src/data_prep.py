import os
import shutil
import warnings

import numpy as np
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)


class ImgPrep:

    def __init__(
        self,
        dir_input,
        dir_output,
        size,
        augmentation_rules,
        train_size=None,
        val_size=None,
        ext_output=".jpg",
    ):
        self.dir_input = dir_input
        self.dir_output = dir_output
        self.size = size
        self.augmentation_rules = augmentation_rules
        self.train_size = train_size
        self.val_size = val_size
        self.ext_output = ext_output
        if train_size and val_size:
            self.test_size = 1 - (train_size + val_size)

    def check_valid(self, img_path, error=False, rm=True, resize_if_small=False):
        """
        Vérifie qu'une image est valide :
        Extension acceptable
        Taille minimale suffisante (≥ self.size)
        Convertit en .jpg si besoin

        Paramètres :
        - img_path (str) : chemin de l'image à vérifier
        - error (bool) : si True raise une erreur au lieu de supprimer silencieusement
        - rm (bool) : si True retire l'image initiale après conversion
        - resize_if_small (bool) : si True, resize les images trop petites au lieu de les rejeter (interpolation bilinéaire)

        Raises:
        - ValueError: Si error=True et image invalide (sans resize)
        """

        ext = os.path.splitext(img_path)[-1].lower()

        if ext in [".ini", ".svg", ".ds_store"]:
            if error:
                raise ValueError(f"Casse toi avec ton {ext} mdrrr")
            else:
                os.remove(img_path)
                return

        try:
            with Image.open(img_path) as img:
                w, h = img.size

                if w < self.size[0] or h < self.size[1]:
                    if resize_if_small:
                        img_rgb = img.convert("RGB")
                        img_resized = img_rgb.resize(
                            self.size, Image.Resampling.BILINEAR
                        )

                        new_file_path = os.path.splitext(img_path)[0] + self.ext_output
                        img_resized.save(new_file_path, format="JPEG", quality=95)

                        if rm and ext != self.ext_output:
                            os.remove(img_path)

                        return

                    else:
                        if error:
                            raise ValueError(
                                f"L'image est trop petite, elle doit faire au moins {self.size}"
                            )
                        else:
                            os.remove(img_path)
                            return

                if ext != self.ext_output:
                    img_rgb = img.convert("RGB")
                    new_file_path = os.path.splitext(img_path)[0] + self.ext_output
                    img_rgb.save(new_file_path, format="JPEG", quality=95)

                    if rm:
                        os.remove(img_path)

        except ValueError:
            raise
        except Exception as e:
            if error:
                raise
            else:
                print(f"Erreur sur {img_path}: {e}")

    def split_dataset(self):
        """
        (Fonction que j'ai récupéré de mon projet sur le CNN du pokedex lol on recycle)
        Sépare un dataset d'audios en trois sous-dossiers : train, val et test.

        Arborescence initiale :
        pokemon/
        │
        ├── abra/
        │   ├── [...].jpg
        │   ├── [...].jpg
        ├── pikachu/
        │...

        Arborescence après séparation :
        pokemon_split/
        │
        ├── train/
        │   ├── abra/
                ├── [...].jpg
                ├── [...].jpg
        │   ├── pikachu/
        │...
        ├── val/
        │   ├── abra/
                ├── [...].jpg
        │   ├── pikachu/
        |...
        ├── test/
        │   ├── abra/
        │       ├── [...].jpg
        │   ├── pikachu/
        │...

        """

        os.makedirs(os.path.join(self.dir_output), exist_ok=True)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.dir_output, split), exist_ok=True)

        for class_dir in os.listdir(self.dir_input):
            class_path = os.path.join(self.dir_input, class_dir)

            if not os.path.isdir(class_path):
                continue

            files = np.array(
                [
                    f
                    for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))
                ]
            )
            np.random.shuffle(files)
            shuff_files = files.tolist()

            n_total = len(shuff_files)
            n_train = int(self.train_size * n_total)
            n_val = int(self.val_size * n_total)

            splits = {
                "train": shuff_files[:n_train],
                "val": shuff_files[n_train : n_train + n_val],
                "test": shuff_files[n_train + n_val :],
            }

            for split, split_files in splits.items():
                split_class_dir = os.path.join(self.dir_output, split, class_dir)
                os.makedirs(split_class_dir, exist_ok=True)
                for fname in split_files:
                    shutil.copy(
                        os.path.join(class_path, fname),
                        os.path.join(split_class_dir, fname),
                    )

    def aug_train(self):

        augmentation = transforms.Compose(
            [
                transforms.RandomRotation(self.augmentation_rules["rotation"]),
                transforms.RandomAffine(
                    degrees=0, translate=self.augmentation_rules["w_h_shift"]
                ),
                transforms.RandomResizedCrop(
                    size=self.size, scale=self.augmentation_rules["zoom"]
                ),
                transforms.RandomHorizontalFlip(
                    p=self.augmentation_rules["horizontal_flip"]
                ),
                transforms.ColorJitter(
                    brightness=self.augmentation_rules["brightness"]
                ),
            ]
        )
        train_dir = os.path.join((self.dir_output), "train")
        pokemons = [
            pokemon
            for pokemon in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, pokemon))
        ]

        for pokemon in pokemons:
            dir_poke = os.path.join(train_dir, pokemon)
            images_pokemon = [
                image
                for image in os.listdir(dir_poke)
                if image.lower().endswith(self.ext_output)
            ]
            originales = images_pokemon.copy()
            compteur = 0

            while len(images_pokemon) < self.augmentation_rules["taille_plancher"]:
                image_choisie = np.random.choice(originales)
                chemin_img_choisie = os.path.join(dir_poke, image_choisie)

                img = Image.open(chemin_img_choisie).convert("RGB")
                img_arti = augmentation(img)

                nv_nom = f"{os.path.splitext(image_choisie)[0]}_aug{compteur}{self.ext_output}"
                save_chm = os.path.join(dir_poke, nv_nom)
                img_arti.save(save_chm)

                compteur += 1
                images_pokemon.append(nv_nom)


if __name__ == "__main__":
    BASE_DIR = os.getenv("BASE_DIR")
    raw_dir = os.path.join(BASE_DIR, "data/raw_data")
    split_dir = os.path.join(BASE_DIR, "data/split_data")
    np.random.seed(77)
    augmentation_rules = {
        "rotation": 10,
        "w_h_shift": (0.1, 0.1),
        "zoom": (0.9, 1.1),
        "horizontal_flip": 0.5,
        "brightness": 0.2,
        "taille_plancher": 100,
    }

    std = ImgPrep(
        dir_input=raw_dir,
        dir_output=split_dir,
        size=(224, 224),
        augmentation_rules=augmentation_rules,
        train_size=0.7,
        val_size=0.2,
    )
    images = [
        os.path.join(raw_dir, pokemon, image)
        for pokemon in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, pokemon))
        for image in os.listdir(os.path.join(raw_dir, pokemon))
        if os.path.isfile(os.path.join(raw_dir, pokemon, image))
    ]
    for image in images:
        std.check_valid(img_path=image)

    std.split_dataset()
    std.aug_train()
