import pickle
import cv2
import numpy as np

from model import EdgeNet


IMAGE_SIZE = 128


# -----------------------------
# Load Model
# -----------------------------

model = EdgeNet()

with open("best_model.pkl", "rb") as f:
    weights = pickle.load(f)

for i, layer in enumerate(model.layers):

    layer.W = weights[f"W{i}"]
    layer.b = weights[f"b{i}"]


# -----------------------------
# Read Image
# -----------------------------

image = cv2.imread("test.jpg")

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

original = image.copy()

image = cv2.resize(
    image,
    (IMAGE_SIZE, IMAGE_SIZE)
)

image = image.astype(np.float32) / 255.0

image = image.transpose(2, 0, 1)

image = np.expand_dims(image, axis=0)


# -----------------------------
# Prediction
# -----------------------------
pred = model.forward(image)

edge = pred[0, 0]

edge = (edge * 255).astype(np.uint8)

edge = cv2.resize(
    edge,
    (original.shape[1], original.shape[0])
)

cv2.imwrite("edge_prediction.png", edge)

print("Saved edge_prediction.png")