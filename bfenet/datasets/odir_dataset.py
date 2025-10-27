import os
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, Dict, Any
from torch.utils.data import Dataset

class ODIRPairedDataset(Dataset):
	def __init__(self, root_dir: str, split: str = "train", image_size: int = 448, resize: int = 512, transforms: bool = True):
		self.root_dir = root_dir
		self.split = split
		self.image_size = image_size
		self.resize = resize
		self.enable_transforms = transforms and split == "train"

		
		if split == "train":
			split_dir = os.path.join(root_dir, "train")
		elif split in ("off_site_test", "on_site_test"):
			split_dir = os.path.join(root_dir, split)
		else:
			raise ValueError("Invalid split")

		self.left_dir = os.path.join(split_dir, "left_images")
		self.right_dir = os.path.join(split_dir, "right_images")
		# prefer images/, fallback to Images/
		images_dir_lower = os.path.join(split_dir, "images")
		images_dir_upper = os.path.join(split_dir, "Images")
		self.images_dir = images_dir_lower if os.path.isdir(images_dir_lower) else images_dir_upper
		self.single_dir = os.path.isdir(self.images_dir)

		# --- 标签文件：支持 csv/xlsx/xls，测试集可无标签 ---
		labels_csv = os.path.join(split_dir, "labels.csv")
		labels_xlsx = os.path.join(split_dir, "labels.xlsx")
		labels_xls = os.path.join(split_dir, "labels.xls")
		labels_path = None
		for p in (labels_csv, labels_xlsx, labels_xls):
			if os.path.exists(p):
				labels_path = p
				break
		if labels_path is not None:
			if labels_path.endswith(".csv"):
				self.labels = pd.read_csv(labels_path)
			else:
				self.labels = pd.read_excel(labels_path)
		else:
			# 强制要求有标签文件以读取列 "Left-Fundus" 和 "Right-Fundus"
			raise FileNotFoundError(f"labels file not found under {split_dir} (expect labels.csv/xlsx/xls)")

		# --- 只接受表格的 "Left-Fundus" / "Right-Fundus" 两列 ---
		required_cols = ["Left-Fundus", "Right-Fundus"]
		cols = set(self.labels.columns.tolist())
		missing = [c for c in required_cols if c not in cols]
		if missing:
			raise KeyError(f"Missing required columns in labels: {missing}. Expect columns: {required_cols}")
		self.image_rows = self.labels[required_cols].values.tolist()

		# 多标签列名：完整或缩写
		full_cols = ["N","DR","G","C","AMD","H","M","O"]
		abbr_cols = ["N","D","G","C","A","H","M","O"]
		self.targets = None
		if self.labels is not None:
			if all(c in self.labels.columns for c in full_cols):
				self.targets = self.labels[full_cols].values.astype(np.float32)
			elif all(c in self.labels.columns for c in abbr_cols):
				self.targets = self.labels[abbr_cols].values.astype(np.float32)

	def __len__(self) -> int:
		return len(self.image_rows)

	def _read_image(self, path: str) -> np.ndarray:
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		if img is None:
			raise FileNotFoundError(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def _center_crop(self, img: np.ndarray, size: int) -> np.ndarray:
		h, w = img.shape[:2]
		start_y = max((h - size) // 2, 0)
		start_x = max((w - size) // 2, 0)
		return img[start_y:start_y+size, start_x:start_x+size]

	def _augment(self, img_l: np.ndarray, img_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		# Random horizontal/vertical flips applied consistently to both eyes
		if np.random.rand() < 0.5:
			img_l = np.flip(img_l, axis=1)
			img_r = np.flip(img_r, axis=1)
		if np.random.rand() < 0.5:
			img_l = np.flip(img_l, axis=0)
			img_r = np.flip(img_r, axis=0)
		# Random resized crop around the center window
		if np.random.rand() < 0.5:
			scale = np.random.uniform(0.8, 1.0)
			crop = int(self.image_size * scale)
			img_l = self._center_crop(img_l, crop)
			img_r = self._center_crop(img_r, crop)
			img_l = cv2.resize(img_l, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
			img_r = cv2.resize(img_r, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
		return img_l, img_r

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		left_name, right_name = self.image_rows[idx]

		def _resolve(pname: str) -> str:
			# 若给的是绝对/相对路径且存在，直接用；否则按目录规则拼接
			if isinstance(pname, str) and (os.path.isabs(pname) or os.path.sep in pname) and os.path.exists(pname):
				return pname
			if self.single_dir:
				return os.path.join(self.images_dir, os.path.basename(pname))
			# fallback: 左右分目录
			return pname

		left_path = _resolve(left_name)
		right_path = _resolve(right_name)
		if not self.single_dir:
			left_path = os.path.join(self.left_dir, left_path) if not os.path.isabs(left_path) else left_path
			right_path = os.path.join(self.right_dir, right_path) if not os.path.isabs(right_path) else right_path
		img_l = self._read_image(left_path)
		img_r = self._read_image(right_path)

		# Resize -> center crop to 448
		img_l = cv2.resize(img_l, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
		img_r = cv2.resize(img_r, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
		img_l = self._center_crop(img_l, self.image_size)
		img_r = self._center_crop(img_r, self.image_size)

		if self.enable_transforms:
			img_l, img_r = self._augment(img_l, img_r)

		# to tensor [0,1] and normalize like ImageNet
		img_l = img_l.astype(np.float32) / 255.0
		img_r = img_r.astype(np.float32) / 255.0
		mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
		img_l = (img_l - mean) / std
		img_r = (img_r - mean) / std

		# CHW
		img_l = np.transpose(img_l, (0, 1, 2))
		img_r = np.transpose(img_r, (0, 1, 2))
		# Fix to CHW
		img_l = np.moveaxis(img_l, -1, 0)
		img_r = np.moveaxis(img_r, -1, 0)

		sample = {
			"left": img_l,
			"right": img_r,
		}
		if self.targets is not None:
			sample["target"] = self.targets[idx]
		return sample
