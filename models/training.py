import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score

import imageutils
from mymodels import TransDataset
from torch.optim import lr_scheduler
from pathlib import Path

def create_dataloaders(df_train, df_valid, df_test, ycol, batch_size):

    sample_image_path = df_train["filepath"].iloc[0]
    img_size = imageutils.get_image_size(sample_image_path)
    print("H*W: ", img_size)
    
    label_encoder = LabelEncoder()
    df_train[ycol] = label_encoder.fit_transform(df_train[ycol])
    df_valid[ycol] = label_encoder.transform(df_valid[ycol])
    df_test[ycol] = label_encoder.transform(df_test[ycol])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    train_dataset = TransDataset(df_train, img_size, ycol, transform=train_transform)
    valid_dataset = TransDataset(df_valid, img_size, ycol, transform=valid_transform)
    test_dataset = TransDataset(df_test, img_size, ycol, transform=valid_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def train_model(model, train_loader, valid_loader, model_save_directory, num_epochs=25, lr=1e-5, weight_decay=1e-4):

    train_losses = []
    valid_losses = []
    train_f1s = []
    valid_f1s = []
    train_aucs = []
    valid_aucs = []
    train_accuracies = []
    valid_accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = None
    best_val_file = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
            optimizer.zero_grad() # 移動幅を０に設定している
            outputs = model(inputs)
            
            #print("outputs: ", outputs.shape)
            #print("outputs.squeeze(): ", outputs.squeeze().shape)
            #print("labels: ", labels.shape)
            
            loss = criterion(outputs.squeeze(), labels)
            loss.backward() # 偏微分の実行。評価のときは実行しない。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 移動幅が大きすぎたときにクリップして調整する
            optimizer.step() # 偏微分に基づいて実際に移動する。評価のときは移動しない。評価のときは学習しないから。
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_f1 = f1_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])
        epoch_acc = accuracy_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])

        train_losses.append(epoch_loss)
        train_f1s.append(epoch_f1)
        train_accuracies.append(epoch_acc)

        #print(f'Epoch {epoch}/{num_epochs - 1} | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f} | AUC: {epoch_auc:.4f}')
        
        model.eval() # Validation だから。
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad(): # Withの内側では偏微分をしません。
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels) # バッチのデフォルトは平均値がかえってくる。
                # no_grad()があるかないかで挙動が変わる。勾配計算(=偏微分)のための情報を保持しない。メモリ使用量が変わる。
                # バッチサイズは学習と評価で違ってOK
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            best_val_file = Path(model_save_directory) / f"model_{epoch}.pt"
            torch.save(model.state_dict(), best_val_file)
        
        #print(f"[VAL] val_loss: {val_loss}")
        val_loss /= len(valid_loader.dataset)
        #print(f"[VAL] val_loss(div): {val_loss}")

        val_f1 = f1_score(val_labels, [1 if x >= 0 else 0 for x in val_preds])
        val_acc = accuracy_score(val_labels, [1 if x >= 0 else 0 for x in val_preds])

        valid_losses.append(val_loss)
        valid_f1s.append(val_f1)
        valid_accuracies.append(val_acc)

        print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}')

    torch.save(model.state_dict(), Path(model_save_directory) / f"model_last.pt")
    
    plt.figure(figsize=(14, 10))
    epochs = range(num_epochs)
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, valid_f1s, label='Valid F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return best_val_file

def test_classification_report(df):

    print(classification_report(df["labels"], df["predictions"]))
    
    tones = df["skin tone"].unique()
    for t in tones:
        subset = df[df["skin tone"] == t]
        accuracy = accuracy_score(subset["labels"], subset["predictions"])
        print(f"Skin tone {t}: Accuracy {accuracy}")



def train_model_weight(model, train_loader, valid_loader, model_save_directory, num_epochs=25, lr=1e-5, weight_decay=1e-4):

    train_losses = []
    valid_losses = []
    train_f1s = []
    valid_f1s = []
    train_aucs = []
    valid_aucs = []
    train_accuracies = []
    valid_accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = None
    best_val_file = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
            optimizer.zero_grad() # 移動幅を０に設定している
            outputs = model(inputs)
            
            #print("outputs: ", outputs.shape)
            #print("outputs.squeeze(): ", outputs.squeeze().shape)
            #print("labels: ", labels.shape)
            
            loss = criterion(outputs.squeeze(), labels)
            loss.backward() # 偏微分の実行。評価のときは実行しない。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 移動幅が大きすぎたときにクリップして調整する
            optimizer.step() # 偏微分に基づいて実際に移動する。評価のときは移動しない。評価のときは学習しないから。
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_f1 = f1_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])
        epoch_acc = accuracy_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])

        train_losses.append(epoch_loss)
        train_f1s.append(epoch_f1)
        train_accuracies.append(epoch_acc)

        #print(f'Epoch {epoch}/{num_epochs - 1} | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f} | AUC: {epoch_auc:.4f}')
        
        model.eval() # Validation だから。
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad(): # Withの内側では偏微分をしません。
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels) # バッチのデフォルトは平均値がかえってくる。
                # no_grad()があるかないかで挙動が変わる。勾配計算(=偏微分)のための情報を保持しない。メモリ使用量が変わる。
                # バッチサイズは学習と評価で違ってOK
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            best_val_file = Path(model_save_directory) / f"model_{epoch}.pt"
            torch.save(model.state_dict(), best_val_file)
        
        #print(f"[VAL] val_loss: {val_loss}")
        val_loss /= len(valid_loader.dataset)
        #print(f"[VAL] val_loss(div): {val_loss}")

        val_f1 = f1_score(val_labels, [1 if x >= 0 else 0 for x in val_preds])
        val_acc = accuracy_score(val_labels, [1 if x >= 0 else 0 for x in val_preds])

        valid_losses.append(val_loss)
        valid_f1s.append(val_f1)
        valid_accuracies.append(val_acc)

        print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}')

    torch.save(model.state_dict(), Path(model_save_directory) / f"model_last.pt")
    
    plt.figure(figsize=(14, 10))
    epochs = range(num_epochs)
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, valid_f1s, label='Valid F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return best_val_file
