import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from pathlib import Path

def calculate_distance_penalty(penalty_model, distances):
    diff = penalty_model.predict(np.array(distances).reshape(-1, 1))
    diff = 1 - diff
    diff = torch.tensor(diff)
    return diff

class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, penalty_model):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.penalty_model = penalty_model
    def forward(self, outputs, targets, distances, device, epoch, total_epochs, start): # criterion
        outputs = outputs.view(-1)
        bce_loss = F.binary_cross_entropy(torch.sigmoid(outputs), targets, reduction="none") # reduction="none"１つずつ距離を適用させるため
        penalty = calculate_distance_penalty(self.penalty_model, distances).to(device)

        if epoch <= start:
            return bce_loss.mean(), bce_loss.mean()
        else:
            #alpha = 0.95
            alpha = 1 # CelebA and UTKface Resnet, HAM VGG
            penalty = torch.softmax(penalty, dim=0)
            return bce_loss.mean(), (bce_loss * penalty * alpha).sum()

class ModelTrainer:
    def __init__(self, penalty_model, model_save_directory):
        self.penalty_model = penalty_model
        self.model_save_directory = model_save_directory

    def train(self, model, train_loader, valid_loader, start, num_epochs=25, lr=1e-5, weight_decay=1e-4):
        
        train_losses = []
        valid_losses = []
        train_f1s = []
        valid_f1s = []
        train_aucs = []
        valid_aucs = []
        train_accuracies = []
        valid_accuracies = []
        train_losses_nop = []
        valid_losses_nop = []

        best_val_loss = None
        best_val_file = None
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        criterion = CustomBCEWithLogitsLoss(self.penalty_model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_loss_nop = 0.0
            all_preds = []
            all_labels = []
            
            for inputs, labels, distances in train_loader:
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_nop, loss = criterion(outputs.squeeze(), labels, distances, device, epoch, num_epochs, start)
                loss.backward() # Partial Derivative
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_loss_nop += loss_nop.item() * inputs.size(0)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_nop = running_loss_nop / len(train_loader.dataset)
            epoch_f1 = f1_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])
            epoch_auc = roc_auc_score(all_labels, all_preds)
            epoch_acc = accuracy_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])
    
            train_losses.append(epoch_loss)
            train_losses_nop.append(epoch_loss_nop)
            train_f1s.append(epoch_f1)
            train_aucs.append(epoch_auc)
            train_accuracies.append(epoch_acc)
            
            model.eval() # Validation
            val_loss = 0.0
            val_loss_nop = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels, distances in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                    outputs = model(inputs)
                    loss_nop, loss = criterion(outputs.squeeze(), labels, distances, device, epoch, num_epochs, start)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_loss_nop += loss_nop.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            if best_val_loss is None or best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_file = Path(self.model_save_directory) / f"model_{epoch}.pt"
                torch.save(model.state_dict(), best_val_file)
            
            val_loss /= len(valid_loader.dataset)
            val_loss_nop /= len(valid_loader.dataset)
            val_f1 = f1_score(val_labels, [1 if x >= 0.5 else 0 for x in val_preds])
            val_auc = roc_auc_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, [1 if x >= 0.5 else 0 for x in val_preds])
    
            valid_losses.append(val_loss)
            valid_losses_nop.append(val_loss_nop)
            valid_f1s.append(val_f1)
            valid_aucs.append(val_auc)
            valid_accuracies.append(val_acc)
    
            print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
    
        epochs = range(num_epochs)
        plt.figure(figsize=(20, 4))

        fontsize=9
        
        plt.subplot(1, 5, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, valid_losses, label='Valid Loss')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        plt.title('Loss', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.subplot(1, 5, 2)
        plt.plot(epochs, train_losses_nop, label='Train Loss no penalty')
        plt.plot(epochs, valid_losses_nop, label='Valid Loss no penalty')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss no penalty', fontsize=fontsize)
        plt.title('Loss no penalty', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.subplot(1, 5, 3)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.title('Accuracy', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.subplot(1, 5, 4)
        plt.plot(epochs, train_f1s, label='Train F1 Score')
        plt.plot(epochs, valid_f1s, label='Valid F1 Score')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('F1 Score', fontsize=fontsize)
        plt.title('F1 Score', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.subplot(1, 5, 5)
        plt.plot(epochs, train_aucs, label='Train AUC Score')
        plt.plot(epochs, valid_aucs, label='Valid AUC Score')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('AUC Score', fontsize=fontsize)
        plt.title('AUC Score', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()
    
        return best_val_file

    def evaluate(self, model, valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model.eval()
        with torch.no_grad():
            test_predictions = []
            test_outputs = []
            for inputs, labels, distance in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                outputs = model(inputs)
                test_outputs.extend(outputs)
                test_predictions.extend(1 if x >= 0 else 0 for x in outputs)

        test_outputs = [o.cpu().item() for o in test_outputs]
        return test_predictions, test_outputs

    def report(self, df):
        print(classification_report(df["labels"], df["predictions"]))
    
        tones = df["skin tone"].unique()
        for t in tones:
            subset = df[df["skin tone"] == t]
            accuracy = accuracy_score(subset["labels"], subset["predictions"])
            print(f"Skin tone {t}: Accuracy {accuracy}")
        
