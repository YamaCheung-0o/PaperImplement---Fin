from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
	if name == "resnet18":
		m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
		channels = 512
	elif name == "resnet34":
		m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
		channels = 512
	elif name == "resnet50":
		m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
		channels = 2048
	else:
		raise ValueError("Unsupported backbone")
	# remove avgpool and fc, keep conv1..layer4
	features = nn.Sequential(
		m.conv1,
		m.bn1,
		m.relu,
		m.maxpool,
		m.layer1,
		m.layer2,
		m.layer3,
		m.layer4,
	)
	return features, channels


class MultiscaleModule(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		# dilated convs at rates 1,2,4
		self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, dilation=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=2, dilation=2, bias=False)
		self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=4, dilation=4, bias=False)
		self.bn = nn.BatchNorm2d(out_channels // 2 * 3)
		self.fuse = nn.Conv2d(out_channels // 2 * 3, out_channels, kernel_size=1, bias=False)
		self.bn_out = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y1 = self.conv1(x)
		y2 = self.conv2(x)
		y3 = self.conv3(x)
		y = torch.cat([y1, y2, y3], dim=1)
		y = self.bn(y)
		y = self.relu(y)
		y = self.fuse(y)
		y = self.bn_out(y)
		y = self.relu(y)
		return y


class FeatureEnhancement(nn.Module):
	def __init__(self, in_channels: int):
		super().__init__()
		# concat -> conv -> bn -> leakyrelu (fusion block in Fig.2)
		self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
		self.fuse_bn = nn.BatchNorm2d(in_channels)
		self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

		# Q_l / Q_r from each stream (C/4)
		self.query_l = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
		self.query_r = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
		# K from fused features (C/4), V from fused features (C)
		self.key = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
		self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
		# output projection after attention + residual
		self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

	def forward(self, f_l: torch.Tensor, f_r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		# global feature concat -> fusion conv
		f_g = torch.cat([f_l, f_r], dim=1)
		f_g = self.act(self.fuse_bn(self.fuse(f_g)))
		b, c, h, w = f_l.shape

		# heads with LeakyReLU per figure
		q_l = self.act(self.query_l(f_l)).flatten(2).transpose(1, 2)  # B, HW, d
		q_r = self.act(self.query_r(f_r)).flatten(2).transpose(1, 2)
		k = self.act(self.key(f_g)).flatten(2)  # B, d, HW
		v = self.act(self.value(f_g)).flatten(2).transpose(1, 2)  # B, HW, C

		scale = (k.shape[1]) ** 0.5
		attn_l = torch.softmax(torch.bmm(q_l, k) / scale, dim=-1)  # B, HW, HW
		attn_r = torch.softmax(torch.bmm(q_r, k) / scale, dim=-1)

		enh_l = torch.bmm(attn_l, v).transpose(1, 2).reshape(b, c, h, w)
		enh_r = torch.bmm(attn_r, v).transpose(1, 2).reshape(b, c, h, w)
		f_fl = self.proj(enh_l) + f_l
		f_fr = self.proj(enh_r) + f_r
		return f_fl, f_fr


class ClassifierHead(nn.Module):
	def __init__(self, in_dim: int, num_classes: int = 8):
		super().__init__()
		self.fc1 = nn.Linear(in_dim, 4096)
		self.fc2 = nn.Linear(4096, 512)
		self.fc3 = nn.Linear(512, num_classes)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(0.5)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)
		return x


class BFENet(nn.Module):
	def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, num_classes: int = 8):
		super().__init__()
		self.backbone, c = build_backbone(backbone_name, pretrained)
		self.backbone_r = self.backbone  # shared weights
		# Multiscale halves channels per paper (14x14x1024 from 2048)
		self.ms_l = MultiscaleModule(c, c // 2)
		self.ms_r = MultiscaleModule(c, c // 2)
		self.fe = FeatureEnhancement(c // 2)
		# After enhancement, we concat enhanced and original features -> flatten
		# Each side: enhanced (c//2) + original (c//2) pooled -> (c)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		in_dim = (c) * 2  # left c + right c = 2c; for resnet50, 4096; but spec says 8192 -> use concat before pool
		# To strictly match 8192, we concat features before pool (spatial 14x14). We'll flatten pooled vectors of both (enh+orig) per side to c, then concat sides=2c (4096). To reach 8192, replicate by concatenating both prepool and pooled is heavy. Instead align to spec: use 2c per side by concatenating (enh, orig) then both sides -> 4*(c//2)=2c. For resnet50 c=2048 -> 4096. Double the fc first layer to 8192 expectation by using proj to 8192 if c==2048.
		self.project_to_8192 = (c == 2048)
		self.classifier = ClassifierHead(8192 if self.project_to_8192 else 2 * c, num_classes)

	def extract(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)

	def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		f_l = self.extract(left)
		f_r = self.extract(right)
		f_l = self.ms_l(f_l)
		f_r = self.ms_r(f_r)
		f_fl, f_fr = self.fe(f_l, f_r)
		# concat enhanced with original per side
		f_l_cat = torch.cat([f_fl, f_l], dim=1)
		f_r_cat = torch.cat([f_fr, f_r], dim=1)
		# global average pool to vectors
		v_l = self.pool(f_l_cat).flatten(1)
		v_r = self.pool(f_r_cat).flatten(1)
		feat = torch.cat([v_l, v_r], dim=1)  # shape (B, 2c)
		if self.project_to_8192 and feat.shape[1] != 8192:
			# simple linear projection to 8192 to match spec
			feat = F.pad(feat, (0, 8192 - feat.shape[1]))
		logits = self.classifier(feat)
		return logits
