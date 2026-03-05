import os
import shutil
import random
import sys

TRAIN_DIR_NAME = 'Train'
TEST_DIR_NAME = 'Test'

# Move images from split folders back to root before re-splitting
def merge_folders(data_path):
    possible_folders = [
        TRAIN_DIR_NAME, TEST_DIR_NAME, 
    ]
    
    folders_to_check = [os.path.join(data_path, f) for f in possible_folders]
    existing_folders = [d for d in folders_to_check if os.path.exists(d)]

    if not existing_folders:
        return 
        
    print(f"Found existing split folders: {[os.path.basename(f) for f in existing_folders]}. Merging back...")

    for source_dir in existing_folders:
        if os.path.isdir(source_dir):
            for class_name in os.listdir(source_dir):
                class_src = os.path.join(source_dir, class_name)
                if not os.path.isdir(class_src): continue
                
                class_dst = os.path.join(data_path, class_name)
                os.makedirs(class_dst, exist_ok=True)
                
                for f in os.listdir(class_src):
                    try:
                        shutil.move(os.path.join(class_src, f), os.path.join(class_dst, f))
                    except shutil.Error:
                        pass 
                
                try:
                    os.rmdir(class_src)
                except OSError:
                    pass
            
            try:
                os.rmdir(source_dir)
            except OSError:
                print(f"Warning: Could not remove {source_dir}, it might not be empty.")

    print("Merge complete. Ready to re-split.")

# Split dataset into Train and Test folders by ratio
def split_dataset(data_path, split_ratio=0.8):
    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist.")
        return

    # Reset any previous splits
    merge_folders(data_path)

    # Find all class folders, exclude split directories
    excluded_names = [TRAIN_DIR_NAME, TEST_DIR_NAME, 'train_set', 'test_set']
    
    classes = [d for d in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, d)) 
               and d not in excluded_names]

    if not classes:
        print("No class folders found.")
        return

    print(f"Splitting {len(classes)} classes into '{TRAIN_DIR_NAME}' ({split_ratio*100}%) and '{TEST_DIR_NAME}'...")

    train_dir = os.path.join(data_path, TRAIN_DIR_NAME)
    test_dir = os.path.join(data_path, TEST_DIR_NAME)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    total_moved = 0

    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Shuffle and split images
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        for img in train_imgs:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
            
        for img in test_imgs:
            shutil.move(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))
            
        total_moved += len(images)
        
        try:
            os.rmdir(class_path)
        except OSError:
            pass

    print(f"Done! Processed {total_moved} images.")
    print(f"New locations: {train_dir} and {test_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_test_split.py <path_to_dataset>")
    else:
        split_dataset(sys.argv[1])