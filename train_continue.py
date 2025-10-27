import os
import math
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from bfenet.datasets.odir_dataset import ODIRPairedDataset
from bfenet.models.bfenet import BFENet
from bfenet.utils.scheduler import PolyLR
from bfenet.utils.metrics import compute_metrics

# 添加早停机制类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--data_root", type=str, required=False)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="预训练模型路径，用于继续训练")  # 新增参数
    p.add_argument("--resume_epoch", type=int, default=50, help="从哪个epoch开始继续训练")  # 新增参数
    return p.parse_args()

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# 添加预测分析函数
def analyze_predictions(all_t, all_p, num_classes):
    """分析预测结果，检查类别分布和预测偏向"""
    print("\n=== 预测分析 ===")
    
    # 检查真实标签分布
    true_labels = np.argmax(all_t, axis=1) if all_t.shape[1] > 1 else all_t.flatten()
    true_dist = np.bincount(true_labels, minlength=num_classes)
    print("真实标签分布:", true_dist)
    
    # 检查预测分布
    pred_labels = np.argmax(all_p, axis=1) if all_p.shape[1] > 1 else (all_p > 0.5).astype(int).flatten()
    pred_dist = np.bincount(pred_labels, minlength=num_classes)
    print("预测标签分布:", pred_dist)
    
    # 检查预测概率的统计信息
    print("预测概率统计 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
        np.mean(all_p), np.std(all_p), np.min(all_p), np.max(all_p)))
    
    # 如果所有预测都偏向一个类别，发出警告
    if len(np.unique(pred_labels)) == 1:
        print("⚠️  警告: 所有样本都被预测为同一个类别!")
        print(f"   所有预测都是类别 {np.unique(pred_labels)[0]}")
    
    return true_labels, pred_labels

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.num_workers:
        cfg["num_workers"] = args.num_workers
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = torch.cuda.is_available()

    # Data
    train_set = ODIRPairedDataset(cfg["data_root"], split="train", image_size=cfg["image_size"], resize=cfg["resize"], transforms=True)
    
    # Simple split 80/20 inside train for train/val per paper
    n_total = len(train_set)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(train_set, [n_train, n_val])

    train_loader = DataLoader(train_subset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=use_pin)
    val_loader = DataLoader(val_subset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=use_pin)
    print(f"[Data] total={n_total} train={n_train} val={n_val} batch_size={cfg['batch_size']}")

    # Model
    model = BFENet(backbone_name=cfg["backbone"], pretrained=cfg.get("pretrained", True), num_classes=cfg["num_classes"]).to(device)

    # Loss/Optim/Sched
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["base_lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    max_iter = math.ceil(cfg["epochs"] * len(train_loader))
    scheduler = PolyLR(optimizer, max_iter=max_iter, base_lr=cfg["base_lr"], power=cfg["poly_power"]) 
    print(f"[Optim] base_lr={cfg['base_lr']} momentum={cfg['momentum']} weight_decay={cfg['weight_decay']} poly_power={cfg['poly_power']} max_iter={max_iter}")

    # 新增：加载预训练模型逻辑
    start_epoch = 50
    best_final = -1.0
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"加载预训练模型: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        # 加载模型权重
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        
        # 尝试加载优化器状态
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("已加载优化器状态")
        
        # 设置起始epoch
        start_epoch = checkpoint.get("epoch", args.resume_epoch)
        best_final = checkpoint.get("metrics", {}).get("final", -1.0)
        
        print(f"从 epoch {start_epoch} 开始继续训练，之前最佳final分数: {best_final:.3f}")
        
        # 调整学习率调度器
        # if "scheduler" in checkpoint:
        #     scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        if args.checkpoint:
            print(f"警告: 预训练模型文件不存在 {args.checkpoint}，从头开始训练")
        print("从头开始训练")

    # 早停机制
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=cfg.get("min_delta", 0.001))
    
    # Logging
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_continued.ckpt")

    def _to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    # 训练历史记录
    train_losses = []
    val_metrics_history = []

    print(f"\n开始训练，总epochs: {cfg['epochs']}, 起始epoch: {start_epoch}")
    
    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n[Epoch {epoch+1}/{cfg['epochs']}] start")
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar, start=1):
            left = _to_tensor(batch["left"]).to(device)
            right = _to_tensor(batch["right"]).to(device)
            target = _to_tensor(batch["target"]).to(device)

            optimizer.zero_grad()
            logits = model(left, right)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(lr=f"{current_lr:.6f}", loss=f"{loss.item():.4f}")

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch+1}] train_done avg_loss={avg_train_loss:.4f}")

        # Validate
        print(f"[Epoch {epoch+1}] validate...")
        model.eval()
        all_t = []
        all_p = []
        val_loss = 0.0
        
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val {epoch+1}", leave=False)
            for batch in vbar:
                left = _to_tensor(batch["left"]).to(device)
                right = _to_tensor(batch["right"]).to(device)
                target = batch["target"]
                if isinstance(target, torch.Tensor):
                    target_np = target.cpu().numpy()
                else:
                    target_np = target
                
                logits = model(left, right)
                loss = criterion(logits, _to_tensor(target).to(device))
                val_loss += loss.item()
                
                prob = torch.sigmoid(logits).cpu().numpy()
                all_t.append(target_np)
                all_p.append(prob)
        
        all_t = np.concatenate(all_t, axis=0)
        all_p = np.concatenate(all_p, axis=0)
        metrics = compute_metrics(all_t, all_p)
        
        # 预测分析（仅在需要时）
        if epoch == start_epoch or metrics['kappa'] == 0 or metrics['f1'] == 0:
            true_labels, pred_labels = analyze_predictions(all_t, all_p, cfg["num_classes"])
        
        avg_val_loss = val_loss / len(val_loader)
        val_metrics_history.append(metrics)
        
        print(f"[Epoch {epoch+1}] metrics: Kappa={metrics['kappa']:.3f} F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} final={metrics['final']:.3f} val_loss={avg_val_loss:.4f}")

        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("早停触发！停止训练")
            break

        # 保存最佳模型
        if metrics["final"] > best_final:
            best_final = metrics["final"]
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "cfg": cfg,
                "val_loss": avg_val_loss
            }, best_path)
            print(f"[Checkpoint] new best final={best_final:.3f} saved -> {best_path}")

    print(f"\n训练完成！最佳 Final-score: {best_final:.3f} -> {best_path}")
    
    # 输出训练总结
    print("\n=== 训练总结 ===")
    print(f"总训练轮次: {len(train_losses)}")
    print(f"最佳验证分数: {best_final:.3f}")
    print(f"最终模型保存位置: {best_path}")

if __name__ == "__main__":
    main()