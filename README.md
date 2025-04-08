# GeoAI: Semantic Segmentation for Earth from Above

GeoAI is a deep learning project for semantic segmentation of satellite imagery. It focuses on two key environmental challenges: identifying water bodies (including flood zones) and detecting deforested areas. The idea is to automate the kind of manual tagging work that's traditionally slow, labor-intensive, and inconsistent — using convolutional neural networks instead.

---

## What It Does

### Water Body & Flood Detection  
This module uses a U-Net with a ResNet50 encoder to segment water regions in satellite images. It’s trained to distinguish between water and non-water pixels with a decent level of precision, even in complex, high-resolution imagery.

### Deforestation Detection  
This part uses a deeper, more experimental U-Net variant with custom skip connections and multiple upsampling paths. It segments forested vs. cleared land, handling class imbalance with a hybrid loss function. The model’s tuned to pick up on subtle visual differences that might be hard for traditional rule-based systems.

---

## Dataset Note

The dataset used for training isn’t included in this repository due to size constraints, but it’s publicly available and easy to download. You’ll just need to follow the format used in the project and place your images and masks accordingly. A few tweaks in the config, and you’re good to go.

---

## Why This Project Exists

Because we have a lot of satellite images and not enough annotated masks.  
Because environmental monitoring deserves better tools.  

---

## Credits

This project builds on the work of several published papers around U-Net, hybrid loss functions, and segmentation for remote sensing. It’s also powered by a lot of trial and error. So obviously it is not perfect, but I hope to make it much better some day.

