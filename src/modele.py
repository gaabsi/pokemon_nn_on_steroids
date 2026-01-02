import os

import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import DeiTForImageClassification, DeiTImageProcessor

torch.set_float32_matmul_precision("medium")


class PkmnDataset(Dataset):
    """ """

    def __init__(self, data_dir, split, feature_extractor):

        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.feature_extractor = feature_extractor
        self.classes = sorted(
            [
                pokemon
                for pokemon in os.listdir(self.split_dir)
                if os.path.isdir(os.path.join(self.split_dir, pokemon))
            ]
        )
        self.class_index = {pokemon: indx for indx, pokemon in enumerate(self.classes)}
        self.files = []
        self.labels = []

        for pokemon in self.classes:

            pokemon_dir = os.path.join(self.split_dir, pokemon)
            label = self.class_index[pokemon]

            for image in os.listdir(pokemon_dir):
                if ".jpg" in image:
                    self.files.append(os.path.join(pokemon_dir, image))
                    self.labels.append(label)

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        chemin_image = self.files[index]
        label = self.labels[index]

        image = Image.open(chemin_image).convert("RGB")

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        image = inputs["pixel_values"].squeeze(0)

        return image, label


class ModeleEntrainement(pl.LightningModule):
    """ """

    def __init__(self, nb_classes, lr=1e-4):

        super().__init__()
        self.save_hyperparameters()
        self.model = DeiTForImageClassification.from_pretrained(
            "facebook/deit-base-distilled-patch16-224",
            num_labels=nb_classes,
            ignore_mismatched_sizes=True,
        )
        self.nb_classes = nb_classes
        self.lr = lr

        for param in self.model.deit.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Linear(
            self.model.config.hidden_size, self.nb_classes
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        return self.model(x).logits

    def training_step(self, batch, batch_index):

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_index):

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class DataModule(pl.LightningDataModule):
    """ """

    def __init__(self, data_dir, batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.feature_extractor = DeiTImageProcessor.from_pretrained(
            "facebook/deit-base-distilled-patch16-224"
        )

    def setup(self, stage=None):

        self.train_dataset = PkmnDataset(self.data_dir, "train", self.feature_extractor)
        self.val_dataset = PkmnDataset(self.data_dir, "val", self.feature_extractor)
        self.test_dataset = PkmnDataset(self.data_dir, "test", self.feature_extractor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    BASE_DIR = os.getenv("BASE_DIR")
    DATA_DIR = os.path.join(BASE_DIR, "data/split_data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    pl.seed_everything(77, workers=True)

    version = 0
    batch_size = 64
    epochs = 10
    patience = 5
    nb_classes = len(
        [
            pokemon
            for pokemon in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, pokemon))
        ]
    )

    models_dir = os.path.join(BASE_DIR, "models", f"version_{version}")
    os.makedirs(models_dir, exist_ok=True)

    datamodule = DataModule(data_dir=DATA_DIR, batch_size=batch_size)
    model = ModeleEntrainement(nb_classes=nb_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=models_dir,
            filename=f"version_{version}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=models_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        precision="16-mixed",
    )

    trainer.fit(model=model, datamodule=datamodule)
