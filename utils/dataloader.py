import os
import random

import cv2
import numpy as np
from scipy.io import loadmat


class BSDS500Loader:

    def __init__(
        self,
        image_dir,
        gt_dir,
        image_size=128,
        batch_size=8,
        shuffle=True
    ):

        self.image_dir = image_dir
        self.gt_dir = gt_dir

        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.images = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.endswith(".jpg")
            ]
        )

    ##########################################################

    def __len__(self):
        return len(self.images)

    ##########################################################

    def __iter__(self):

        indices = list(range(len(self.images)))

        if self.shuffle:
            random.shuffle(indices)

        for start in range(
            0,
            len(indices),
            self.batch_size
        ):

            batch = indices[start:start + self.batch_size]

            X = []
            Y = []

            for idx in batch:

                ############################################
                # IMAGE
                ############################################

                img_name = self.images[idx]

                img_path = os.path.join(
                    self.image_dir,
                    img_name
                )

                img = cv2.imread(img_path)

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                img = cv2.resize(
                    img,
                    (self.image_size,
                     self.image_size)
                )

                img = img.astype(np.float32) / 255.0

                # HWC -> CHW
                img = img.transpose(2, 0, 1)

                ############################################
                # GROUND TRUTH
                ############################################

                mat_name = img_name.replace(
                    ".jpg",
                    ".mat"
                )

                mat_path = os.path.join(
                    self.gt_dir,
                    mat_name
                )

                mat = loadmat(mat_path)

                ground_truth = mat["groundTruth"]

                num_annotations = ground_truth.shape[1]

                edge = np.zeros(
                    ground_truth[0][0]["Boundaries"][0][0].shape,
                    dtype=np.float32
                )

                for i in range(num_annotations):

                    boundary = ground_truth[0][i]["Boundaries"][0][0]

                    edge += boundary.astype(np.float32)

                # Average annotations
                edge /= num_annotations

                # Binary edge map
                edge = (edge > 0.3).astype(np.float32)

                edge = cv2.resize(
                    edge,
                    (self.image_size,
                     self.image_size),
                    interpolation=cv2.INTER_LINEAR
                )

                edge = np.expand_dims(
                    edge,
                    axis=0
                )

                X.append(img)
                Y.append(edge)

            X = np.asarray(
                X,
                dtype=np.float32
            )

            Y = np.asarray(
                Y,
                dtype=np.float32
            )

            yield X, Y