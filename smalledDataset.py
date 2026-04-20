import os
import shutil

data_dir = "ImageNet/"
output_dir = "ImageNetSmall/"

IMAGES_PER_CLASS = 35

classes = sorted(os.listdir(data_dir))

selected_images = []
ground_truth = []

for class_idx, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    images = sorted(os.listdir(class_path))
    chosen = images[:IMAGES_PER_CLASS]
    
    for img in chosen:
        src = os.path.join(class_path, img)
        dst = os.path.join(output_dir, class_name, img)
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)  # ✅ COPY (not move)
        
        selected_images.append(os.path.join(class_name, img))
        ground_truth.append(class_idx)

# save ground truth
with open("validation_ground_truth_subset.txt", "w") as f:
    for label in ground_truth:
        f.write(str(label) + "\n")

print("Done. Images copied, original dataset untouched.")