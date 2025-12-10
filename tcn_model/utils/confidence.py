# utils/confidence.py

import torch
import torch.nn.functional as F

def compute_confidence_score(logits):
    """
    حساب درجة الثقة من مخرجات النموذج (logits).
    نستخدم softmax لأخذ أعلى احتمالية كدرجة للثقة.
    """
    probabilities = F.softmax(logits, dim=1)
    confidence_scores, _ = torch.max(probabilities, dim=1)
    return confidence_scores
