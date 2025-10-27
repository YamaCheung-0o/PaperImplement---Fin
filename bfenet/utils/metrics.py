from typing import Dict
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	# y_true: (N, C) multi-label 0/1
	# y_prob: (N, C) probabilities
	y_pred = (y_prob >= threshold).astype(np.int32)
	# Kappa for patient-level multi-label: average across classes
	kappas = []
	f1s = []
	aucs = []
	for c in range(y_true.shape[1]):
		try:
			kappas.append(cohen_kappa_score(y_true[:, c], y_pred[:, c]))
		except Exception:
			kappas.append(0.0)
		try:
			f1s.append(f1_score(y_true[:, c], y_pred[:, c]))
		except Exception:
			f1s.append(0.0)
		try:
			aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
		except Exception:
			aucs.append(0.0)
	kappa = float(np.mean(kappas))
	f1 = float(np.mean(f1s))
	auc = float(np.mean(aucs))
	final = float((kappa + f1 + auc) / 3.0)
	return {"kappa": kappa, "f1": f1, "auc": auc, "final": final}
