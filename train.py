import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LensDataset
from model import LensClassifier
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def train_model(epochs=5, batch_size=64, lr=1e-3):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dir = '/Users/mywishanand/Documents/1. Multi-Class Classification/dataset/train'
    val_dir = '/Users/mywishanand/Documents/1. Multi-Class Classification/dataset/val'
    
    train_dataset = LensDataset(train_dir)
    val_dataset = LensDataset(val_dir)
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = LensClassifier(num_classes=3, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}] (Train)')
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch [{epoch+1}/{epochs}] (Val)'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved Best Model!")

if __name__ == "__main__":
    train_model(epochs=3, batch_size=128)
