import torch
from torch.utils.data import DataLoader
from dataset import LensDataset
from model import LensClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_dir = '/Users/mywishanand/Documents/1. Multi-Class Classification/dataset/val'
    val_dataset = LensDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = LensClassifier(num_classes=3, in_channels=1)
    
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print("Model file not found! Train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    all_labels = []
    all_probs = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1) # get probabilities
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    classes = ['no', 'sphere', 'vort']
    
    plt.figure(figsize=(10, 8))
    
    for i, cls_name in enumerate(classes):
        # Create binary labels for one-vs-rest evaluation
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'Class {cls_name} (AUC = {roc_auc:.4f})')
        print(f"AUC for {cls_name}: {roc_auc:.4f}")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("Saved ROC curve to roc_curve.png")

if __name__ == "__main__":
    evaluate_model()
