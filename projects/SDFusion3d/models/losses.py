# src/losses.py
import torch

def combined_loss(task_loss, sparse_weight, dense_weight, lambda_reg=0.1):
    reg_loss = lambda_reg * torch.sum(torch.abs(sparse_weight - dense_weight))
    total_loss = task_loss + reg_loss
    return total_loss
