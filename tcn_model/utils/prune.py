# utils/prune.py

import torch
import torch.nn.utils.prune as prune

def auto_prune_model(model, pruning_ratio=0.5):
    """
    تنفيذ Auto-Pruning على كل طبقة Conv1d في النموذج.
    نحذف النسبة المحددة من الفلاتر بناءً على الحجم.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
            prune.remove(module, "weight")  # إزالة القناع بعد الحذف

    return model
