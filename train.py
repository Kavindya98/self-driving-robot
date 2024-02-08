import os
import time
import shutil
import logging
import traceback

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader
import torchvision.models
from torchvision import transforms

from model import BaseModel
from model import AlexNet
from model import ResNet18Enc, ResNet18_Pretrained, ViT_Pretrained
from dataset import RvssDataset
from config import parse_args


LOSS_KEYS = ["loss", "l1_loss"]


def train(args, logging, result_dir):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    logging.info("model loaded...")
    model = ViT_Pretrained()
    model = model.to(device)
    logging.info(model)

    rvsd = RvssDataset(args.train_data, hist_len=1)
    data_loader = DataLoader(rvsd, batch_size=args.batch_size, shuffle=True, num_workers=2)

    rvsdt = RvssDataset(args.test_data, hist_len=1)
    test_loader = DataLoader(rvsdt, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    logging.info("training...")
    model.train()
    loss_train = {}
    loss_valid = {}
    for ep in range(args.epochs):
        losses_batch = {}
        # training loop
        model.train()
        for idx, (batch_im, batch_act) in enumerate(data_loader):

            # transform = transforms.Compose([transforms.ConvertImageDtype(torch.float),
            #                                 transforms.Normalize(mean=MEAN,std=STD)])
            # batch_im = transform(batch_im)
            batch_im = batch_im.to(device)
            batch_act = batch_act.to(device)

            output = model(batch_im.float().squeeze(dim=1).permute(0, 3, 1, 2))
            losses = model.get_loss(output, batch_act)
            #loss = losses["weight_l2_loss"]
            loss = losses["l1_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in LOSS_KEYS:
                losses_batch[k] = losses_batch.get(k, []) + [losses[k].item()]
            logging.info(
                "[TRAIN] epoch: {}/{}, iter: {}/{}, loss: {:4f}, l1 loss: {:.4f}".format(
                    ep, args.epochs, idx, len(data_loader),
                    losses["loss"].item(),
                    losses["l1_loss"].item()
                )
            )
        for k in LOSS_KEYS:
            loss_train[k] = loss_train.get(k, []) + [np.mean(losses_batch["loss"])]
        
        # validation loop
        model.eval()
        losses_batch = {}
        for idx, (batch_im, batch_act) in enumerate(test_loader):
            batch_im = batch_im.to(device)
            batch_act = batch_act.to(device)
            output = model(batch_im.float().squeeze(dim=1).permute(0, 3, 1, 2))
            losses = model.get_loss(output, batch_act)
            loss = losses["loss"]

            for k in LOSS_KEYS:
                losses_batch[k] = losses_batch.get(k, []) + [losses[k].item()]
        for k in LOSS_KEYS:
            loss_valid[k] = loss_valid.get(k, []) + [np.mean(losses_batch[k])]

        logging.info(
                "[EVAL] epoch: {}/{}, iter: {}/{}, loss: {:4f}, l1 loss: {:.4f}".format(
                    ep, args.epochs, idx, len(test_loader),
                    loss_valid["loss"][-1],
                    loss_valid["l1_loss"][-1]
                )
            )

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(1, 2)
        plt.subplot(gs[0, 0])
        plt.plot(loss_train["loss"])
        plt.title("Train loss")

        plt.subplot(gs[0, 1])
        plt.plot(loss_valid["loss"])
        plt.title("Valid loss")

        plt.savefig("{}/pics/pic{}.png".format(result_dir, ep))
        plt.close()
        torch.save(model, "{}/models/checkpoint{}.pt".format(result_dir, ep))

def main():
    train(args, logging, result_dir)

if __name__ == "__main__":
    try:
        args = parse_args()
        if args.debug:
            ttag = "test"
        else:
            ttag = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        result_dir = "results/{}".format(ttag)

        if args.debug and os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs("{}/models".format(result_dir), exist_ok=True)
        os.makedirs("{}/pics".format(result_dir), exist_ok=True)
        os.makedirs("{}/logs".format(result_dir), exist_ok=True)

        logging.basicConfig(
            filename='{}/logs/log.txt'.format(result_dir),
            level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%Y%m%d_%H%M%S'
        )
        
        logging.info("message: {}".format(args.message))
        main()
    except Exception as e:
        logging.error(traceback.format_exc())

# difference between diffusion model and variational auto encoder, probablisitc graph model and variational inferences.
# high order difusion models richard hartley
# find high density geodiesic pahts in diffusion latent space
# weighted pathway weighed by the probabilist
# caluculas of variation.
# diffusion is the solution to the shortest path problem???
