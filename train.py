import pickle
import os
import numpy as xp

from model import EdgeNet
from losses.bce import BinaryCrossEntropy
from optim.sgd import SGD
from utils.dataloader import BSDS500Loader


# -------------------------
# Hyperparameters
# -------------------------

EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 8
IMAGE_SIZE = 128


# -------------------------
# Dataset
# -------------------------

train_loader = BSDS500Loader(
    image_dir="dataset/images/train",
    gt_dir="dataset/groundTruth/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = BSDS500Loader(
    image_dir="dataset/images/val",
    gt_dir="dataset/groundTruth/val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = EdgeNet()

criterion = BinaryCrossEntropy()

optimizer = SGD(
    model.layers,
    lr=LR
)

best_val = float("inf")


# -------------------------
# Training
# -------------------------

for epoch in range(EPOCHS):

    ####################################
    # TRAIN
    ####################################

    train_loss = 0

    train_batches = 0

    for X, Y in train_loader:
        X = xp.asarray(X)
        Y = xp.asarray(Y)

        pred = model.forward(X)

        loss = criterion.forward(pred, Y)

        grad = criterion.backward()

        model.backward(grad)

        optimizer.step()

        optimizer.zero_grad()

        train_loss += loss

        train_batches += 1

    train_loss /= train_batches

    ####################################
    # VALIDATION
    ####################################

    val_loss = 0

    val_batches = 0

    for X, Y in val_loader:
        X = xp.asarray(X)
        Y = xp.asarray(Y)

        pred = model.forward(X)

        loss = criterion.forward(pred, Y)

        val_loss += loss

        val_batches += 1

    val_loss /= val_batches

    ####################################
    # SAVE BEST MODEL
    ####################################

    if val_loss < best_val:

        best_val = val_loss

        weights = {}

        for i, layer in enumerate(model.layers):

            weights[f"W{i}"] = layer.W

            weights[f"b{i}"] = layer.b

        with open("best_model.pkl", "wb") as f:

            pickle.dump(weights, f)

    ####################################

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train {train_loss:.5f} | "
        f"Val {val_loss:.5f}"
    )


print("\nTraining Finished.")