# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from compressai.datasets import ImageFolder
from compressai.zoo import models
import numpy as np
import matplotlib.pyplot as plt
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
        capturable=True
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
        capturable=True
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device


    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)


        '''
        print("out_net shape ", out_net['x_hat'].shape)

        fig, axes = plt.subplots(1, 9, figsize=(40, 16))
        in_img = d.to('cpu')
        in_img = in_img.detach().numpy()
        # Loop through each image and plot it
        for i in range(9):
            axes[i].imshow(in_img[0][i], cmap="gray")
            axes[i].set_title(i)

            axes[i].axis("off")

        plt.tight_layout()



        fig, axes = plt.subplots(1, 9, figsize=(40, 16))
        out_img = out_net['x_hat'].to('cpu')
        out_img = out_img.detach().numpy()
        # Loop through each image and plot it
        for i in range(9):
            axes[i].imshow(out_img[0][i], cmap="gray")
            axes[i].set_title(i)

            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
        exit()
        '''


        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="cnn",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0070,

        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256,256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [ transforms.ToTensor(),transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model]()
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        remove_prefix ='module.'
        checkpoint["state_dict"] = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint["state_dict"].items()}
        '''[ checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not any(bad_name in k for bad_name in ["g_a.0.weight", "g_s.8.weight", "g_s.8.bias"])
        }'''

        #print("keys isn the ckpt ;",checkpoint["state_dict"])

        # Initialize specified weights with Xavier initialization
        #checkpoint["state_dict"]["g_a.0.weight"] = init.xavier_normal_(torch.empty([192, 9, 5, 5]))
        #checkpoint["state_dict"]["g_s.8.weight"] = init.xavier_normal_(torch.empty([192, 9, 5, 5]))
        #checkpoint["state_dict"]["g_s.8.bias"] = torch.zeros(net.state_dict()["g_s.8.bias"].shape[0])

        # Freeze all other layers except the specified ones
        """for param_name, param in net.named_parameters():
            if (param_name.startswith('g_a.1.weight') or param_name.startswith('g_a.2.weight') or param_name.startswith('g_a.3.weight') or param_name.startswith('g_a.4.weight')
                or param_name.startswith('g_a.5.weight') or param_name.startswith('g_a.6.weight') or param_name.startswith('g_a.7.weight') or param_name.startswith('g_s.1.weight')
                or param_name.startswith('g_s.2.weight') or param_name.startswith('g_s.3.weight') or param_name.startswith('g_s.4.weight') or param_name.startswith('g_s.5.weight')
                or param_name.startswith('g_s.7.weight') or  param_name.startswith('g_s.6.weight')):
                param.requires_grad = False
            else:
                param.requires_grad = True"""



        #last_epoch = checkpoint["epoch"]

        net.load_state_dict(checkpoint["state_dict"])

        #print(net)
        #print("optimizer_parameters:",checkpoint["optimizer"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                args.save_path,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
