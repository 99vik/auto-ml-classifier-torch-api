from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import json
import torch

def train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations, progress_queue, labels, device):
    accuracy = Accuracy(task='multiclass', num_classes=len(labels)).to(device)
    f1score = F1Score(task="multiclass", num_classes=len(labels)).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(labels)).to(device)

    for i in range(iterations + 1):
        model.train()
        logits = model(X_tr)
        logits_pred = torch.softmax(logits, dim=1).argmax(dim=1)
        acc = accuracy(logits_pred, y_tr).item()*100

        loss = loss_fn(logits, y_tr)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        model.eval()
        if (i) % (iterations / 20) == 0:
            with torch.inference_mode():
                test_logits = model(X_te)
                test_logits_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                test_loss = loss_fn(test_logits, y_te)
                test_acc = accuracy(test_logits_pred, y_te).item()*100
                f1 = f1score(test_logits_pred, y_te)
                confusion_matrix = confmat(test_logits_pred, y_te)

            training_data = {
                 "status": "training",
                 "iteration": str(i),
                 "trainLoss": loss.item(),
                 "trainAccuracy": acc,
                 "testLoss": test_loss.item(),
                 "testAccuracy": test_acc,
                 "f1Score": f1.item(),
                 "confusionMatrix": confusion_matrix.tolist()
            }
            progress_queue.put(json.dumps(training_data))