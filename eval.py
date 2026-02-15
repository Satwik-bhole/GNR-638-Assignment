import sys
import os
import my_dl_framework as nn

BATCH_SIZE=32

def load_model_weights(layers,load_path):
    if not os.path.exists(load_path):
        print(f"Error: Directory {load_path} not found.")
        sys.exit(1)
    print(f"Loading weights from {load_path}...")
    for i,layer in enumerate(layers):
        if hasattr(layer,'weights'):
            w_path=os.path.join(load_path,f"layer_{i}_{type(layer).__name__}_weights.bin")
            if os.path.exists(w_path):layer.weights.load(w_path)
            else:print(f"Warning: {w_path} not found!")
        if hasattr(layer,'bias'):
            b_path=os.path.join(load_path,f"layer_{i}_{type(layer).__name__}_bias.bin")
            if os.path.exists(b_path):layer.bias.load(b_path)
            else:print(f"Warning: {b_path} not found!")
    print("Weights loaded.")

def main(test_path,load_path):
    print(f"Loading TEST data from {test_path}...")
    dataset=nn.load_dataset(test_path)
    test_imgs=dataset.images
    test_lbls=dataset.labels
    conv1=nn.Conv2D(3,16,3,1,1)
    relu1=nn.ReLU()
    pool1=nn.MaxPool2D(2,2)
    conv2=nn.Conv2D(16,32,3,1,1)
    relu2=nn.ReLU()
    pool2=nn.MaxPool2D(2,2)
    conv3=nn.Conv2D(32,64,3,1,1)
    relu3=nn.ReLU()
    pool3=nn.MaxPool2D(2,2)
    flat=nn.Flatten()
    fc=nn.Linear(64*4*4,10)
    layers=[conv1,relu1,pool1,conv2,relu2,pool2,conv3,relu3,pool3,flat,fc]
    load_model_weights(layers,load_path)
    correct=0
    total=len(test_imgs)
    print("Starting Evaluation...")
    for i in range(0,total,BATCH_SIZE):
        batch_imgs_raw=test_imgs[i:i+BATCH_SIZE]
        batch_lbls=test_lbls[i:i+BATCH_SIZE]
        x=nn.batch_tensors(batch_imgs_raw)
        out=x
        for l in layers:out=l.forward(out)
        logits=out.data
        for b in range(len(batch_imgs_raw)):
            sample_logits=logits[b*10:(b+1)*10]
            pred=sample_logits.index(max(sample_logits))
            if pred==batch_lbls[b]:correct+=1
        if(i//BATCH_SIZE)%10==0:
            print(f"Processed {min(i+BATCH_SIZE,total)} / {total} images...")
    acc=correct/total
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python eval.py <path_to_test_data> <path_to_load_weights>")
    else:
        main(sys.argv[1],sys.argv[2])