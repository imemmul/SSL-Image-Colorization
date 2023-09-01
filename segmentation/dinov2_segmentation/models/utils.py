import torch
import numpy as np


def compute_iou(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask)
    union = torch.logical_or(pred_mask, true_mask)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou

def compute_miou(predictions, targets, num_classes):
    iou_scores = []
    
    for class_idx in range(num_classes):
        pred_class = predictions == class_idx
        target_class = targets == class_idx

        iou = compute_iou(pred_class, target_class)
        iou_scores.append(iou.item())

    miou = np.mean(iou_scores)
    return miou


def compute_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    accuracy = correct / total
    return accuracy

def compute_mean_accuracy(predictions, targets, num_classes):
    accuracy_scores = []

    for class_idx in range(num_classes):
        pred_class = predictions == class_idx
        target_class = targets == class_idx

        accuracy = compute_accuracy(pred_class, target_class)
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    return mean_accuracy
