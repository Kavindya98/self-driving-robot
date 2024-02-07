import time

import torch
from torch.utils.data import DataLoader

from model import BaseModel
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


    model = BaseModel()
    print("model loaded...")

    rvsd = RvssDataset(args.data_folder, hist_len=1)
    data_loader = DataLoader(rvsd, batch_size=2, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(),
        lr=1e-3,
        weight_decay=0
    )

    print("training...")
    for ep in range(epochs):
        for idx, (batch_im, batch_act) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(batch_im.float())

            loss = model.get_loss(output, batch_act)

            loss.backward()
            optimizer.step()

            print(
                "epoch: {}/{}, iter: {}/{}, loss: {:4f}".format(
                    ep, epochs, idx, len(data_loader), loss.item()
                )
            )
        model.save("checkpoint{}.pt".format(ep))