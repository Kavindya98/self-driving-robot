import os
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader

from model import BaseModel
from model import AlexNet
from model import ResNet18Enc
from dataset import RvssDataset
from config import parse_args

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        ttag = "test"
    else:
        ttag = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    result_dir = "results/{}".format(ttag)

    if args.debug:
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs("{}/models".format(result_dir), exist_ok=True)
    os.makedirs("{}/pics".format(result_dir), exist_ok=True)
    

    print("{} message: {}".format(ttag, args.message))
    epochs = args.epochs

    print("model loaded...")
    model = ResNet18Enc()
    print(model)

    rvsd = RvssDataset(args.data_folder, hist_len=1)
    data_loader = DataLoader(rvsd, batch_size=32, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(),
        lr=1e-4,
        weight_decay=0
    )

    print("training...")
    loss_stats = []
    for ep in range(epochs):
        losses_batch = {}
        for idx, (batch_im, batch_act) in enumerate(data_loader):
            output = model(batch_im.float().squeeze(dim=1).permute(0, 3, 1, 2))
            losses = model.get_loss(output, batch_act)

            loss = losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_batch["loss"] = losses_batch.get("loss", []) + [losses["loss"].item()]

            print(
                "epoch: {}/{}, iter: {}/{}, loss: {:4f}, l1 loss: {:.4f}".format(
                    ep, epochs, idx, len(data_loader), 
                    losses["loss"].item(),
                    losses["l1_loss"].item()
                )
            )
        loss_stats.append(np.mean(losses_batch["loss"]))

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(1, 1)

        plt.subplot(gs[0, 0])
        plt.plot(loss_stats)

        plt.savefig("{}/pics/pic{}.png".format(result_dir, ep))
        plt.close()
        torch.save(model, "{}/models/checkpoint{}.pt".format(result_dir, ep))


# difference between diffusion model and variational auto encoder, probablisitc graph model and variational inferences.
# high order difusion models richard hartley
# find high density geodiesic pahts in diffusion latent space
# weighted pathway weighed by the probabilist
# caluculas of variation.
# diffusion is the solution to the shortest path problem???
