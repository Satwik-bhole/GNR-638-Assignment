import sys
import time
import random
import os
import my_dl_framework as nn

BATCH_SIZE = 64
LR = 0.01 
MOMENTUM = 0.9
EPOCHS = 10

def save_model(layers, save_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    print(f"\nSaving weights to {save_path}...")
    for i, layer in enumerate(layers):
        if hasattr(layer, 'weights'):
            layer.weights.save(os.path.join(save_path, f"layer_{i}_{type(layer).__name__}_weights.bin"))
        if hasattr(layer, 'bias'):
            layer.bias.save(os.path.join(save_path, f"layer_{i}_{type(layer).__name__}_bias.bin"))
    print("Save complete.")

def main(train_path, save_path):
    print(f"Loading data from {train_path}...")
    load_start = time.time()
    dataset = nn.load_dataset(train_path)
    num_classes = len(dataset.class_names)
    print(f"Dataset Loading Time: {time.time() - load_start:.4f} seconds") 

    # Model definition [cite: 42-46, 99]
    conv1 = nn.Conv2D(3, 16, 3, 1, 1)    # Reduced from 32
    relu1 = nn.ReLU()
    pool1 = nn.MaxPool2D(2, 2)
    
    conv2 = nn.Conv2D(16, 32, 3, 1, 1)   # Reduced from 64
    relu2 = nn.ReLU()
    pool2 = nn.MaxPool2D(2, 2)
    
    # We remove the 128-filter layer to save massive computation time
    flat = nn.Flatten()
    
    # Update the input size: 32x32 -> Pool1 -> 16x16 -> Pool2 -> 8x8
    # 32 filters * 8 * 8 image size
    fc = nn.Linear(32 * 8 * 8, num_classes) 

    layers = [conv1, relu1, pool1, conv2, relu2, pool2, flat, fc]
    optim = nn.SGD(LR, MOMENTUM) # Momentum implementation from course
    
    indices = list(range(len(dataset.images)))
    metrics_printed = False 

    print(f"Starting Training | Classes: {num_classes}")
    train_start = time.time()

    for epoch in range(EPOCHS):
        random.shuffle(indices)
        epoch_loss = 0.0
        correct = 0
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            x = nn.batch_tensors([dataset.images[k] for k in batch_idx])
            targets = [dataset.labels[k] for k in batch_idx]
            
            # Forward pass
            out = x
            for l in layers: out = l.forward(out)
            
            # Report Efficiency Metrics once dimensions are known [cite: 47, 48, 101]
            if not metrics_printed:
                total_params = sum(l.params for l in layers if hasattr(l, 'params'))
                macs_val = sum(l.macs for l in layers if hasattr(l, 'macs')) // len(batch_idx)
                print("-" * 30)
                print(f"Total Parameters: {total_params}") 
                print(f"MACs per sample: {macs_val}") 
                print("-" * 30)
                metrics_printed = True

            # Loss computation with epsilon fix for 'inf'
            grad_back = nn.Tensor([0], False)
            loss = nn.cross_entropy_loss(out, targets, grad_back)
            epoch_loss += loss * len(batch_idx)
            
            # Backpropagation [cite: 15, 22]
            for l in reversed(layers): grad_back = l.backward(grad_back)

            # Gradient-based optimization [cite: 25]
            optim.step(conv1.weights, conv1.bias)
            optim.step(conv2.weights, conv2.bias)
            optim.step(fc.weights, fc.bias)
            
            # PRINT EVERY 10 BATCHES
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i//BATCH_SIZE} | Loss: {loss:.4f}")

            logits = out.data
            for b in range(len(batch_idx)):
                sample_logits = logits[b * num_classes:(b + 1) * num_classes]
                if sample_logits.index(max(sample_logits)) == targets[b]: 
                    correct += 1

        print(f"Epoch {epoch+1} DONE | Avg Loss: {epoch_loss/len(indices):.4f} | Acc: {correct/len(indices):.4f}") 
    
    print(f"Total Training Time: {time.time() - train_start:.2f}s") 
    save_model(layers, save_path)

if __name__ == "__main__":
    if len(sys.argv) < 3: 
        print("Usage: python train.py <data_path> <save_path>")
    else: 
        main(sys.argv[1], sys.argv[2])