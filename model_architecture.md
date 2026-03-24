# ResNet18 Model Architecture for Lens Classification

![ResNet18 Architecture](/Users/mywishanand/.gemini/antigravity/brain/2431b64b-7287-4198-a30f-16ef34484f22/resnet18_lens_architecture_1774363053356.png)

The model used is a modified **ResNet-18 (Residual Network with 18 layers)**. It is a convolutional neural network (CNN) that uses "skip connections" or "shortcuts" to jump over some layers, which helps avoid the vanishing gradient problem and allows for deeper training.

## Architectural Description

1.  **Input Layer**: Accepts grayscale images of size `(N, 1, 150, 150)`, where `N` is the batch size.
2.  **Modified Entry Convolution (`conv1`)**:
    *   **Original**: 64 filters, `7x7` kernel, stride 2, padding 3 (expects 3-channel RGB).
    *   **Modified**: Changed to accept **1-channel grayscale** input.
3.  **Basic Blocks**: The network consists of 4 main stages, each containing 2 "Basic Blocks".
    *   Each **Basic Block** has two `3x3` convolutional layers.
    *   **Skip Connections**: The input of each block is added to the output of the two convolutions.
4.  **Global Average Pooling**: Reduces the spatial dimensions `(H, W)` to `(1, 1)` per feature map.
5.  **Modified Fully Connected Layer (`fc`)**:
    *   **Original**: Linear layer with 512 input features and 1000 output classes.
    *   **Modified**: Linear layer with **512 input features and 3 output classes** (`no`, `sphere`, `vort`).

## Architectural Map (Mermaid)

```mermaid
graph TD
    Input["Input Image (1x150x150)"] --> Conv1["Conv1 (7x7, Stride 2)"]
    Conv1 --> BN1["Batch Norm + ReLU"]
    BN1 --> MaxPool["Max Pool (3x3, Stride 2)"]
    
    subgraph Stage1["Stage 1 (64 channels)"]
        B1_1["BasicBlock 1"] --> B1_2["BasicBlock 2"]
    end
    MaxPool --> Stage1
    
    subgraph Stage2["Stage 2 (128 channels)"]
        B2_1["BasicBlock 1 (Downsample)"] --> B2_2["BasicBlock 2"]
    end
    Stage1 --> Stage2
    
    subgraph Stage3["Stage 3 (256 channels)"]
        B3_1["BasicBlock 1 (Downsample)"] --> B3_2["BasicBlock 2"]
    end
    Stage2 --> Stage3
    
    subgraph Stage4["Stage 4 (512 channels)"]
        B4_1["BasicBlock 1 (Downsample)"] --> B4_2["BasicBlock 2"]
    end
    Stage3 --> Stage4
    
    Stage4 --> AvgPool["Global Average Pool"]
    AvgPool --> Flatten["Flatten"]
    Flatten --> FC["Fully Connected (Linear 512 -> 3)"]
    FC --> Output["Output Logits (3)"]
```

## Layer Summary Table

| Layer Type | Output Shape | Parameters |
| :--- | :--- | :--- |
| **Input** | `(1, 150, 150)` | - |
| **Conv1 (7x7)** | `(64, 75, 75)` | ~3,136 |
| **Stage 1** | `(64, 38, 38)` | ~147,456 |
| **Stage 2** | `(128, 19, 19)` | ~524,288 |
| **Stage 3** | `(256, 10, 10)` | ~2,097,152 |
| **Stage 4** | `(512, 5, 5)` | ~8,388,608 |
| **Global Avg Pool** | `(512, 1, 1)` | - |
| **FC Layer** | `(3)` | ~1,539 |
