import sys
import time
import random
import os
import my_dl_framework as nn

BATCH_SIZE=32
LR=0.001
EPOCHS=10

def save_model(layers,save_path):
    if not os.path.exists(save_path):os.makedirs(save_path)
    print(f"Saving model to {save_path}...")
    for i,layer in enumerate(layers):
        if hasattr(layer,'weights'):
            w_path=os.path.join(save_path,f"layer_{i}_{type(layer).__name__}_weights.bin")
            layer.weights.save(w_path)
        if hasattr(layer,'bias'):
            b_path=os.path.join(save_path,f"layer_{i}_{type(layer).__name__}_bias.bin")
            layer.bias.save(b_path)
    print("Model saved successfully.")

def main(train_path,save_path):
    print(f"Loading data from {train_path}...")
    
    # --- DATA LOADING TIMER ---
    load_start=time.time()
    dataset=nn.load_dataset(train_path)
    train_imgs=dataset.images
    train_lbls=dataset.labels
    print(f"Data Loading Time: {time.time()-load_start:.4f} seconds")
    # --------------------------

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
    optim=nn.SGD(LR)
    indices=list(range(len(train_imgs)))

    print(f"Starting Training on {len(indices)} images...")
    print(f"Weights will be saved to: {save_path}")

    # --- TRAINING TIMER ---
    train_start=time.time()

    for epoch in range(EPOCHS):
        random.shuffle(indices)
        epoch_loss=0.0
        correct=0
        for i in range(0,len(indices),BATCH_SIZE):
            batch_idx=indices[i:i+BATCH_SIZE]
            batch_imgs_raw=[train_imgs[k] for k in batch_idx]
            batch_lbls=[train_lbls[k] for k in batch_idx]
            x=nn.batch_tensors(batch_imgs_raw)
            out=x
            for l in layers:out=l.forward(out)
            grad_back=nn.Tensor([0],False)
            loss=nn.cross_entropy_loss(out,batch_lbls,grad_back)
            epoch_loss+=loss*len(batch_idx)
            for l in reversed(layers):grad_back=l.backward(grad_back)
            optim.step(conv1.weights,conv1.bias)
            optim.step(conv2.weights,conv2.bias)
            optim.step(conv3.weights,conv3.bias)
            optim.step(fc.weights,fc.bias)
            logits=out.data
            for b in range(len(batch_idx)):
                sample_logits=logits[b*10:(b+1)*10]
                pred=sample_logits.index(max(sample_logits))
                if pred==batch_lbls[b]:correct+=1
            if(i//BATCH_SIZE)%10==0:
                print(f"Epoch {epoch+1} | Batch {i//BATCH_SIZE} | Loss: {loss:.4f}")
        print(f"Epoch {epoch+1} DONE | Loss: {epoch_loss/len(indices):.4f} | Acc: {correct/len(indices):.4f}")
    
    total_time=time.time()-train_start
    print(f"Training Complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes).")
    save_model(layers,save_path)

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python train.py <path_to_data> <path_to_save_weights>")
    else:
        main(sys.argv[1],sys.argv[2])