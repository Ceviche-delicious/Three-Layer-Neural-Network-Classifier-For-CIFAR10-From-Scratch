import os
import sys
import json

sys.path.append("..")

import matplotlib.pyplot as plt

from src.Models import MLPModel, Linear

if not os.path.exists("images"):
    os.makedirs("images")

ckpt_path = "./models_01/model_epoch_200.pkl"
nn_architecture = json.load(open(ckpt_path.replace(".pkl", ".json"), "r"))

model = MLPModel(nn_architecture)
j = 0
for i, layer in enumerate(model.layers):
    if isinstance(layer, Linear):
        j = j + 1
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {j} Initial Weight Distribution")
        plt.savefig(f"images/layer_{j}_weight_distribution_init.png")

        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {j} Initial Weight Matrix")
        plt.colorbar()
        plt.savefig(f"images/layer_{j}_weight_matrix_init.png")

model.load_model_dict(path=ckpt_path)
j = 0
for i, layer in enumerate(model.layers):
    if isinstance(layer, Linear):
        j = j + 1
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {j} Weight Distribution After Training")
        plt.savefig(f"images/layer_{j}_weight_distribution.png")

        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {j} Weight Matrix After Training")
        plt.colorbar()
        plt.savefig(f"images/layer_{j}_weight_matrix.png")