import os
import torch
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from datasets.FASDataset import FASDataset
from utils.utils import read_cfg, get_optimizer, get_device, build_network
from trainer.FASTrainer import FASTrainer
from models.loss import DepthPPGLoss, DepthLoss

cfg = read_cfg(cfg_file="config/CDCN_adam_lr1e-3.yaml")
#cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
#cfg = read_cfg(cfg_file="config/C_CDN_adam.yaml")
#cfg = read_cfg(cfg_file="config/DC_CDN_adam.yaml")

test_csv = "high_all_test.csv"
#test_csv = "high_30k_test.csv"
#test_csv = "high_20k_test.csv"

device = get_device(cfg)

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

#criterion = DepthPPGLoss(device=device)
criterion = DepthLoss(device=device)

writer = SummaryWriter(cfg['log_dir'])

val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=test_csv,
    depth_map_size=cfg['model']['depth_map_size'],
    transform=val_transform,
    smoothing=cfg['train']['smoothing']
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=cfg['val']['batch_size'],
    shuffle=True,
    num_workers=2
)

trainer = FASTrainer(
    cfg=cfg, 
    network=network,
    optimizer=optimizer,
    criterion=criterion,
    lr_scheduler=None,
    device=device,
    trainloader=[],
    valloader=valloader,
    writer=writer
)

trainer.load_model("CDCN_CelebA_Spoof_e12_acc_0.9441.pth")

val_acc = trainer.validate(0)
print("val acc: %.4f"%val_acc)
