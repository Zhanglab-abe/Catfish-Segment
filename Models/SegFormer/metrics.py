# import torch
# import numpy as np

# def pixel_accuracy(output, target):
#     with torch.no_grad():
#         _, output = torch.max(output, dim=1)
#         correct = torch.sum(output == target).item()
#         total = target.numel()
#     return correct / total

# def mean_iou(output, target, num_classes):
#     with torch.no_grad():
#         output = torch.argmax(output, dim=1).squeeze(1).cpu().numpy()
#         target = target.squeeze(1).cpu().numpy()
#         iou_list = []
#         for cls in range(num_classes):
#             intersection = np.logical_and(output == cls, target == cls).sum()
#             union = np.logical_or(output == cls, target == cls).sum()
#             if union == 0:
#                 iou = float('nan')  # if there is no ground truth, do not include in evaluation
#             else:
#                 iou = intersection / union
#             iou_list.append(iou)
#         m_iou = np.nanmean(iou_list)
#     return m_iou

# def accuracy(output, target):
#     with torch.no_grad():
#         output = torch.argmax(output, dim=1)
#         correct = torch.sum(output == target).item()
#         total = target.numel()
#     return correct / total

import torch
import numpy as np

def pixel_accuracy(output, target):
    with torch.no_grad():
        _, output = torch.max(output, dim=1)
        correct = torch.sum(output == target).item()
        total = target.numel()
    return correct / total

def mean_iou(output, target, num_classes):
    with torch.no_grad():
        # output = torch.argmax(output, dim=1).squeeze(1).cpu().numpy()
        output = torch.argmax.logits.detach().cpu().numpy()
        output = np.argmax(pred_mask, axis=1)
        target = target.squeeze(1).cpu().numpy()
        iou_list = []
        for cls in range(num_classes):
            intersection = np.logical_and(output == cls, target == cls).sum()
            union = np.logical_or(output == cls, target == cls).sum()
            if union == 0:
                iou = float('nan')  # if there is no ground truth, do not include in evaluation
            else:
                iou = intersection / union
            iou_list.append(iou)
        m_iou = np.nanmean(iou_list)
    return m_iou

def accuracy(output, target):
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        correct = torch.sum(output == target).item()
        total = target.numel()
    return correct / total

def mean_accuracy(output, target, num_classes):
    with torch.no_grad():
        output = torch.argmax(output, dim=1).squeeze(1).cpu().numpy()
        target = target.squeeze(1).cpu().numpy()
        acc_list = []
        for cls in range(num_classes):
            correct = np.sum((output == cls) & (target == cls))
            total = np.sum(target == cls)
            if total == 0:
                acc = float('nan')  # if there is no ground truth, do not include in evaluation
            else:
                acc = correct / total
            acc_list.append(acc)
        m_acc = np.nanmean(acc_list)
    return m_acc

def average_accuracy(output, target, num_classes):
    with torch.no_grad():
        output = torch.argmax(output, dim=1).squeeze(1).cpu().numpy()
        target = target.squeeze(1).cpu().numpy()
        acc_list = []
        for cls in range(num_classes):
            correct = np.sum((output == cls) & (target == cls))
            total = np.sum(output == cls)
            if total == 0:
                acc = float('nan')  # if there is no prediction, do not include in evaluation
            else:
                acc = correct / total
            acc_list.append(acc)
        a_acc = np.nanmean(acc_list)
    return a_acc


