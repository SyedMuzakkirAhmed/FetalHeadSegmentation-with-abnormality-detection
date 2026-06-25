# Transformer-Enhanced U-Net for Fetal Ultrasound Analysis

## Overview

This project presents an automated fetal ultrasound analysis pipeline using a Transformer-Enhanced U-Net segmentation model and a ResNet50-based classification model. The system performs fetal head segmentation, biometric parameter extraction, gestational age estimation, and abnormality classification from ultrasound images.

The segmentation model identifies the fetal head region, after which geometric analysis is performed to calculate clinically important fetal biometric parameters.

---

## Features

### Fetal Head Segmentation

* Transformer-Enhanced U-Net architecture
* Encoder-decoder network with attention mechanisms
* Automatic fetal head mask generation
* BCE + Dice Loss optimization

### Biometric Parameter Extraction

* Head Circumference (HC)
* Biparietal Diameter (BPD)
* Occipitofrontal Diameter (OFD)
* Head Area

### Gestational Age Estimation

Gestational Age (weeks) ≈ HC (mm) / 10

### Abnormality Classification

* ResNet50-based classifier
* Binary classification:

  * Malignant
  * Non-Malignant

---

## Project Pipeline

```text
Ultrasound Image
        ↓
Preprocessing & Resize
        ↓
Transformer-Enhanced U-Net
        ↓
Mask Prediction
        ↓
Contour Extraction
        ↓
Ellipse Fitting
        ↓
HC / BPD / OFD / Area Calculation
        ↓
Pixel-to-mm Conversion
        ↓
Gestational Age Estimation
        ↓
Abnormality Classification (ResNet50)
        ↓
Final Results
```

---

## Segmentation Architecture

The segmentation stage uses a Transformer-Enhanced U-Net architecture.

### Encoder

* Convolutional feature extraction
* Transformer blocks for global context learning

### Decoder

* Feature upsampling
* Skip connections
* Pixel-wise segmentation

### Loss Function

Loss = BCE Loss + Dice Loss

Binary Cross Entropy improves pixel classification while Dice Loss improves overlap between predicted and ground truth masks.

---

## Biometric Measurement

After segmentation, ellipse fitting is applied to the extracted fetal head contour.

### Head Circumference (HC)

HC ≈ π [3(a+b) − √((3a+b)(a+3b))]

### Biparietal Diameter (BPD)

BPD = 2b

### Occipitofrontal Diameter (OFD)

OFD = 2a

### Head Area

Area = Contour Area

where:

* a = semi-major axis
* b = semi-minor axis

---

## Age Estimation

Gestational age is estimated from the calculated head circumference:

Age (weeks) = HC(mm) / 10

This provides an approximate fetal age estimation.

---

## Abnormality Detection

A ResNet50 model is trained separately using fetal ultrasound images.

### Classification Categories

* Malignant
* Non-Malignant

### Model Features

* Transfer Learning
* Weighted Sampling
* Data Augmentation
* Binary Classification

The model achieved approximately 88% classification accuracy on the validation dataset.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn

---

## Running the Segmentation Pipeline

### Step 1 – Activate Environment

```bash
cd U:\project\tensorf\src
tf210\Scripts\activate
```

### Step 2 – Train the Model

```bash
python TensorflowUNetTrainer.py
```

### Step 3 – Generate Segmentation Masks

```bash
python TensorflowUNetInferencer.py
```

### Step 4 – Calculate Biometric Parameters

```bash
python para.py
```

Generated outputs:

```text
predicted_measurements.csv
measurement_visuals/
```

---

## Running Abnormality Detection

### Train the Classifier

```bash
python abnormal.py
```

### Evaluate the Classifier

```bash
python test_abnormal.py
```

---

## Results

The proposed framework successfully:

* Segments fetal head regions automatically
* Extracts HC, BPD, OFD, and Head Area
* Estimates gestational age
* Performs abnormality classification
* Reduces manual measurement effort
* Provides consistent and reproducible biometric measurements

---

## Future Work

* Real-time ultrasound integration
* Multi-parameter fetal growth assessment
* Direct gestational age regression
* Clinical validation on larger datasets
* Multi-class abnormality classification

---

## Repository Structure

```text
src/
├── TensorflowUNetTrainer.py
├── TensorflowUNetInferencer.py
├── para.py
├── losses.py
├── TensorflowTransUNet.py
├── TensorflowUNet.py
├── abnormal_age/
│   ├── abnormal.py
│   └── test_abnormal.py
├── keras_unet_collection/
└── train_eval_infer.config

generator/
├── ImageMaskDatasetGenerator.py
└── split_master.py
```

---

## Conclusion

This project demonstrates an end-to-end automated fetal ultrasound analysis system that combines Transformer-Enhanced U-Net segmentation, geometric measurement extraction, gestational age estimation, and ResNet50-based abnormality detection. The framework provides a reliable and scalable approach for assisting fetal biometric assessment and prenatal diagnostic workflows.
