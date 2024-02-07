import time

import torch
from torch.utils.data import DataLoader

from model import BaseModel
from model import AlexNet
from dataset import RvssDataset
from config import parse_args

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    args = parse_args()
    print("args parsed...")
    epochs = args.epochs

    model = AlexNet()
    print("model loaded...")

    rvsd = RvssDataset(args.data_folder, hist_len=1)
    data_loader = DataLoader(rvsd, batch_size=32, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(),
        lr=1e-4,
        weight_decay=0
    )

    print("training...")
    for ep in range(epochs):
        for idx, (batch_im, batch_act) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(batch_im.float().squeeze(dim=1).permute(0, 3, 1, 2))
            loss = ((output - batch_act)**2).mean()

            loss.backward()
            optimizer.step()

            print(
                "epoch: {}/{}, iter: {}/{}, loss: {:4f}".format(
                    ep, epochs, idx, len(data_loader), loss.item()
                )
            )
        torch.save(model, "models/checkpoint{}.pt".format(ep))


# difference between diffusion model and variational auto encoder, probablisitc graph model and variational inferences.
# high order difusion models richard hartley
# find high density geodiesic pahts in diffusion latent space
# weighted pathway weighed by the probabilist
# caluculas of variation.
# diffusion is the solution to the shortest path problem???
