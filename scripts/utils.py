def accuracy_fn(y_pred, y_target):
    import torch
    correct = torch.eq(y_pred, y_target).sum().item()
    return (correct/len(y_pred))*100