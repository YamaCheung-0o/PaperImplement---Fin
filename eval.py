import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from bfenet.datasets.odir_dataset import ODIRPairedDataset
from bfenet.models.bfenet import BFENet
from bfenet.utils.metrics import compute_metrics
from tqdm import tqdm  # 导入tqdm


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--config", type=str, default="zyw/PaperImplement/configs/config.yaml")
	p.add_argument("--data_root", type=str, required=True)
	p.add_argument("--checkpoint", type=str, required=True)
	p.add_argument("--split", type=str, choices=["off_site_test", "on_site_test"], required=True)
	return p.parse_args()


def load_config(path: str):
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return cfg


def main():
	args = parse_args()
	cfg = load_config(args.config)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	ds = ODIRPairedDataset(args.data_root, split=args.split, image_size=cfg["image_size"], resize=cfg["resize"], transforms=False)
	loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

	ckpt = torch.load(args.checkpoint, map_location="cpu")
	m_cfg = ckpt.get("cfg", cfg)
	
	model = BFENet(backbone_name=m_cfg["backbone"], pretrained=False, num_classes=m_cfg["num_classes"]).to(device)
	model.load_state_dict(ckpt["state_dict"], strict=True)
	model.eval()

	all_t = []
	all_p = []
	with torch.no_grad():
		for batch in tqdm(loader, desc=f"Processing {args.split}", total=len(loader), ncols=100, unit="batch"):
			left = batch["left"].to(device)
			right = batch["right"].to(device)
			logits = model(left, right)
			prob = torch.sigmoid(logits).cpu().numpy()
			all_p.append(prob)
			if "target" in batch:
				all_t.append(batch["target"])  # numpy already
	if all_t:
		all_t = np.concatenate(all_t, axis=0)
		all_p = np.concatenate(all_p, axis=0)
		metrics = compute_metrics(all_t, all_p)
		print(f"Kappa={metrics['kappa']:.3f} F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} Final={metrics['final']:.3f}")
	else:
		all_p = np.concatenate(all_p, axis=0)
		print(f"Predictions shape: {all_p.shape}. Save with --output if needed.")


if __name__ == "__main__":
	main()
