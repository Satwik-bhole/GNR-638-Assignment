import sys
import os
import time
import my_dl_framework as nn

BATCH_SIZE = 64  # Updated to match train.py

def load_model_weights(layers, load_path):
    if not os.path.exists(load_path):
        print(f"Error: Directory {load_path} not found.")
        sys.exit(1)
    print(f"Loading weights from {load_path}...")
    for i, layer in enumerate(layers):
        if hasattr(layer, 'weights'):
            w_path = os.path.join(load_path, f"layer_{i}_{type(layer).__name__}_weights.bin")
            if os.path.exists(w_path): layer.weights.load(w_path)
        if hasattr(layer, 'bias'):
            b_path = os.path.join(load_path, f"layer_{i}_{type(layer).__name__}_bias.bin")
            if os.path.exists(b_path): layer.bias.load(b_path)
    print("Weights loaded successfully.")

def main(test_path, load_path):
    print(f"Loading TEST data from {test_path}...")
    # [cite_start]Requirement: Measure and report dataset loading time [cite: 39, 40, 102]
    load_start = time.time()
    dataset = nn.load_dataset(test_path)
    num_classes = len(dataset.class_names)
    print(f"Dataset Loading Time: {time.time() - load_start:.4f} seconds")

    # [cite_start]Architecture MUST match train.py exactly [cite: 42-46, 99]
    # Input: 3x32x32
    conv1 = nn.Conv2D(3, 16, 3, 1, 1)    
    relu1 = nn.ReLU()
    pool1 = nn.MaxPool2D(2, 2)           # Output: 16x16x16
    
    conv2 = nn.Conv2D(16, 32, 3, 1, 1)   
    relu2 = nn.ReLU()
    pool2 = nn.MaxPool2D(2, 2)           # Output: 32x8x8
    
    flat = nn.Flatten()
    fc = nn.Linear(32 * 8 * 8, num_classes) 
    
    layers = [conv1, relu1, pool1, conv2, relu2, pool2, flat, fc]

    # [cite_start]Requirement: Report total trainable parameters [cite: 47, 100]
    total_params = sum(l.params for l in layers if hasattr(l, 'params'))
    print(f"Total Model Parameters: {total_params}")

    load_model_weights(layers, load_path)
    
    correct = 0
    total = len(dataset.images)
    print(f"Starting Evaluation on {total} images...")
    
    eval_start = time.time()
    metrics_printed = False

    for i in range(0, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        batch_imgs = [dataset.images[k] for k in range(i, batch_end)]
        batch_lbls = [dataset.labels[k] for k in range(i, batch_end)]
        
        # Requirement: Support batching during evaluation 
        x = nn.batch_tensors(batch_imgs)
        
        out = x
        for l in layers:
            out = l.forward(out)
            
        # [cite_start]Requirement: Print MACs during evaluation [cite: 48, 101]
        if not metrics_printed:
            macs_val = sum(l.macs for l in layers if hasattr(l, 'macs')) // len(batch_imgs)
            print(f"MACs per forward pass (per sample): {macs_val}")
            metrics_printed = True
            
        logits = out.data
        for b in range(len(batch_imgs)):
            sample_logits = logits[b * num_classes : (b + 1) * num_classes]
            pred = sample_logits.index(max(sample_logits))
            if pred == batch_lbls[b]:
                correct += 1
        
        # Simple progress tracking
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"Processed {batch_end}/{total} samples...", end='\r')

    # [cite_start]Requirement: Evaluation metrics [cite: 103]
    acc = correct / total
    print(f"\nEvaluation Time: {time.time() - eval_start:.2f} seconds")
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval.py <test_data_path> <weights_path>")
    else:
        main(sys.argv[1], sys.argv[2])