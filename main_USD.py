# GPL License
# Copyright (C) 2024, Your Institution
# All Rights Reserved
#
# @Time    : 2024/10/17
# @Author  : KJDavid
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from Toolbox.model_USD import UDS2C2
from Toolbox.data_USD import FusionDataset
import numpy as np

# ================== Pre-Define =================== #
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
parser = argparse.ArgumentParser()
parser.add_argument("--lr1", type=float, default=0.0005, help="Learning rate for stage 1")
parser.add_argument("--lr2", type=float, default=0.0005, help="Learning rate for stage 2")
parser.add_argument("--epochs1", type=int, default=1500, help="Number of epochs for stage 1")
parser.add_argument("--epochs2", type=int, default=1500, help="Number of epochs for stage 2")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--device", type=str, default='cuda', help="Device to use")
parser.add_argument("--name", type=str, required=True, help="Data ID")
parser.add_argument("--satellite", type=str, default='m3/', help="Satellite type")
parser.add_argument("--channels_hs", type=int, default=128, help="Hyperspectral channels")
parser.add_argument("--channels_ms", type=int, default=8, help="Multispectral channels")
parser.add_argument("--n", type=int, default=4, help="Spatial downsampling factor")
parser.add_argument("--augment", action='store_true', help="Enable data augmentation")
args = parser.parse_args()

lr1 = args.lr1
lr2 = args.lr2
epochs1 = args.epochs1
epochs2 = args.epochs2
batch_size = args.batch_size
device = torch.device(args.device)
name = args.name
satellite = args.satellite
C = args.channels_hs
C_m = args.channels_ms
n = args.n

print(f"main: n={args.n}, type={type(args.n)}, augment={args.augment}")
model = UDS2C2(C, C_m, n).to(device)
optimizer1 = optim.Adam(model.parameters(), lr=lr1, betas=(0.9, 0.999))
optimizer2 = optim.Adam(model.parameters(), lr=lr2, betas=(0.9, 0.999))

def save_checkpoint(model, name, stage):
    model_out_path = f'model_UDS2C2/{satellite}{name}_stage{stage}.pth'
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train ----------------------------------
###################################################################

def train_stage1(training_data_loader):
    print('Run Stage 1: Training spatial and spectral downsampling...')
    min_loss = float('inf')
    patience = 200
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=20, verbose=True)

    for epoch in range(epochs1):
        model.train()
        epoch_loss = []
        start_epoch = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            X, Y = batch[0].to(device), batch[1].to(device)
            X = X.squeeze(1)
            Y = Y.squeeze(1)
            optimizer1.zero_grad()
            _, Y_l1, Y_l2 = model(X, Y)
            loss1 = torch.mean(torch.abs(Y_l1 - Y_l2))
            epoch_loss.append(loss1.item())
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer1.step()
        t_loss = np.nanmean(np.array(epoch_loss))
        scheduler.step(t_loss)
        if t_loss < min_loss:
            save_checkpoint(model, name, 1)
            min_loss = t_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 1: Early stopping at epoch {epoch}")
            break
        if epoch % 50 == 0:
            print(f'Stage 1: Epoch: {epoch} training loss: {t_loss:.7f}, '
                  f'Time: {time.time() - start_epoch:.2f}s, LR: {optimizer1.param_groups[0]["lr"]:.7f}')

def train_stage2(training_data_loader):
    print('Run Stage 2: Training spectral upsampling...')
    model.load_state_dict(torch.load(f'model_UDS2C2/{satellite}{name}_stage1.pth'))
    min_loss = float('inf')
    patience = 200
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=50, verbose=True)

    for epoch in range(epochs2):
        model.train()
        epoch_loss = []
        start_epoch = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            X, Y = batch[0].to(device), batch[1].to(device)
            X = X.squeeze(1)
            Y = Y.squeeze(1)
            optimizer2.zero_grad()
            Z, _, _ = model(X, Y)
            loss2 = torch.mean(torch.abs(Z - X))
            epoch_loss.append(loss2.item())
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer2.step()
        t_loss = np.nanmean(np.array(epoch_loss))
        scheduler.step(t_loss)
        if t_loss < min_loss:
            save_checkpoint(model, name, 2)
            min_loss = t_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 2: Early stopping at epoch {epoch}")
            break
        if epoch % 50 == 0:
            print(f'Stage 2: Epoch: {epoch} training loss: {t_loss:.7f}, '
                  f'Time: {time.time() - start_epoch:.2f}s, LR: {optimizer2.param_groups[0]["lr"]:.7f}')

###################################################################
# ------------------- Main Function -------------------
###################################################################

if __name__ == "__main__":
    train_set = FusionDataset('../autodl-tmp/m3_128_128_ms_x4', augment=args.augment)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    train_stage1(training_data_loader)
    train_stage2(training_data_loader)
    print('Fusion completed.')