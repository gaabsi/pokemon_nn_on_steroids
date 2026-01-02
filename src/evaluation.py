import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import DeiTForImageClassification, DeiTImageProcessor

from data_prep import ImgPrep
from modele import ModeleEntrainement


class ModelEvalMetriques:
    """
    Classe qui permet de digérer les logs generees par lightning pendant l'entrainement.
    Resume l'entrainement et les performances (ici selon la loss et l'acu) entre le train et le val set.
    A la fin on sauvegarde le plot pour savoir si notre modele a potentiellement overfit, si il devait tourner plus lgtps, ...
    """

    def __init__(self, chemin_logs, chemin_output, nom_figs):
        self.chemin_logs = chemin_logs
        self.chemin_output = chemin_output
        self.nom_figs = nom_figs

    def process_logs(self):
        """
        Fonction de processing des logs de lightning pour avoir quelque chose de plus compact.
        Met tous les logs de notre entrainementdans un pd.Dataframe qui résume train et val accu et loss a chaque fin d'epoch.
        """

        df = pd.read_csv(self.chemin_logs)
        check_condition = ["train_acc_epoch", "val_acc"]
        cols = ["epoch", "train_acc_epoch", "train_loss_epoch", "val_acc", "val_loss"]

        df_logs = (
            df[df[check_condition].notna().any(axis=1)][cols]
            .groupby("epoch")
            .mean()
            .reset_index()
        )

        return df_logs

    def plot_logs(self):
        """
        Fonction pour plot les courbes de loss et d'acc sur les train et val dataset.
        La figure finale sera enregistrée au format png.
        """

        df_logs = self.process_logs()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(
            df_logs["epoch"],
            df_logs["train_loss_epoch"],
            "o-",
            label="Train Loss",
            linewidth=2,
        )
        ax1.plot(
            df_logs["epoch"], df_logs["val_loss"], "s-", label="Val Loss", linewidth=2
        )
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            df_logs["epoch"],
            df_logs["train_acc_epoch"],
            "o-",
            label="Train Acc",
            linewidth=2,
        )
        ax2.plot(
            df_logs["epoch"], df_logs["val_acc"], "s-", label="Val Acc", linewidth=2
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Training & Validation Acc", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.chemin_output, self.nom_figs), bbox_inches="tight"
        )
        plt.close(fig)


class PredictEval:
    """
    Cette classe permet d'effectuer une prediction sur de nouvelles donnees.
    On refait la pipeline dans le sens inverse.
    """

    def __init__(self, checkpoint_path, feature_extractor, dir_classes):
        self.checkpoint_path = checkpoint_path
        self.feature_extractor = DeiTImageProcessor.from_pretrained(feature_extractor)
        self.dir_classes = dir_classes
        self.modele = None
        self.index_nom = None
        self.preprocess = ImgPrep(
            dir_input=None, dir_output=None, size=(224, 224), augmentation_rules=None
        )

    def custom_modele(self):
        """
        Recupere la classe d'entrainement du modele et utilise la classe parente de lightning qui nous permet de load sur un checkpoint.
        Ca nous permet de mettre les poids de notre modele qu'on a custom en amont de cette etape.
        """
        if not self.modele:
            self.modele = ModeleEntrainement.load_from_checkpoint(self.checkpoint_path)
            self.modele.eval()

        return self.modele

    def map_index_nom(self):
        """
        On refait rapidement le mapping index : nom dans un dictionnaire
        On reutilise un bout de code qu'on avait dev dans le modele.py
        """
        if not self.index_nom:
            self.classes = sorted(
                [
                    pokemon
                    for pokemon in os.listdir(self.dir_classes)
                    if os.path.isdir(os.path.join(self.dir_classes, pokemon))
                ]
            )
            self.index_nom = {
                index: pokemon for index, pokemon in enumerate(self.classes)
            }

        return self.index_nom

    def predict(self, img_path, resize=False):
        """
        Ici on fait une fonction qui nous permet de prendre un audio en input et de sortir la classe predite
        On repasse la fonction de preprocessing a notre input comme ça on peut prendre n'importe quel audio en input.
        """

        modele = self.custom_modele()
        index_nom = self.map_index_nom()

        self.preprocess.check_valid(img_path, rm=False, resize_if_small=resize)

        base_path = os.path.splitext(img_path)[0]
        final_path = base_path + ".jpg"

        if os.path.exists(final_path):
            img = Image.open(final_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

        inpoute = self.feature_extractor(images=img, return_tensors="pt")

        impoute_model = inpoute["pixel_values"]

        with torch.no_grad():
            logits = modele(impoute_model)
            probas = torch.softmax(logits, dim=1)

            conf, pred_index = torch.max(probas, dim=1)

            pred_index = pred_index.item()
            conf = conf.item()

        nom_pred = index_nom[pred_index]
        vrai_nom = os.path.dirname(img_path).split("/")[-1]
        correct = int(vrai_nom == nom_pred)

        return nom_pred, vrai_nom, conf, correct


if __name__ == "__main__":

    BASE_DIR = os.getenv("BASE_DIR")
    test_dir = os.path.join(BASE_DIR, "data/split_data/test")
    feature_extractor = "facebook/deit-base-distilled-patch16-224"

    images = [
        os.path.join(test_dir, pokemon, image)
        for pokemon in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, pokemon))
        for image in os.listdir(os.path.join(test_dir, pokemon))
        if os.path.isfile(os.path.join(test_dir, pokemon, image))
    ]

    lightning_dir = os.path.join(BASE_DIR, f"models/version_0/lightning_logs")
    chemin_logs = os.path.join(lightning_dir, "metrics.csv")
    checkpoint_path = os.path.join(BASE_DIR, f"models/version_0/version_0.ckpt")
    test_perf_csv_path = os.path.join(lightning_dir, "test_perf.csv")

    eval_metrics = ModelEvalMetriques(
        chemin_logs=chemin_logs,
        chemin_output=lightning_dir,
        nom_figs="eval_modele.png",
    )
    eval_metrics.plot_logs()

    test_set_eval = PredictEval(
        checkpoint_path=checkpoint_path,
        feature_extractor=feature_extractor,
        dir_classes=test_dir,
    )

    resultats = []
    for image in images:
        try:
            nom_pred, vrai_nom, conf, correct = test_set_eval.predict(image)
            resultats.append(
                {
                    "fichier": os.path.basename(image),
                    "vrai_nom": vrai_nom,
                    "nom_predit": nom_pred,
                    "conf": conf,
                    "correct": correct,
                }
            )
        except Exception as e:
            print(f"Erreur sur {image}: {e}")
            continue

    test_perf = pd.DataFrame(resultats)
    accu = test_perf["correct"].mean()
    test_perf.to_csv(test_perf_csv_path, index=False)

    print(f"Accuracy: {accu:.2%} sur le test set")
