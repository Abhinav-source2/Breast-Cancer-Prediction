
# 🩺 Breast Cancer Detection using Deep Learning  
![Logo](https://img.icons8.com/fluency/96/000000/breast-cancer.png)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19Jhi6cM7LTknwCYVK1xU9V9poDDOKRuz)  
![Python](https://img.shields.io/badge/Python-3.9-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project focuses on detecting breast cancer using deep learning techniques, leveraging two open-source mammography image datasets: **CBIS-DDSM** and a secondary **Breast Cancer MRI dataset**. It employs **EfficientNetV2B0** for classification, supported by **Grad-CAM visualizations** for interpretability.

> **Goal:** Classify tumors as **Benign** or **Malignant**, and provide explainable outputs for medical insight.

---

## 📂 Datasets Used

### 🔹 1. CBIS-DDSM: Curated Breast Imaging Subset of DDSM  
📌 **Source**: [Kaggle Link](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)  
📁 **Size**: ~10,239 mammogram images  
📋 **Modality**: Digital X-ray (Mammography)  
🔍 **Structure**:  
- Two lesion categories:  
  - **Calcification (CALC)**  
  - **Mass (MASS)**  
- **View Types**:  
  - Craniocaudal (CC)  
  - Mediolateral Oblique (MLO)  
- **Labels Included**:  
  - **Pathology**: Benign or Malignant  
  - **BI-RADS score** (1–5): Radiological suspicion level  
  - **Breast density** (1–4)  
  - **ROI Mask**: Defines the lesion location  
- **Format**: DICOM + ROI overlay masks

🧪 **Use in This Project**:
- DICOMs converted to PNG and resized
- Masks used to extract ROI (Region of Interest)
- Used for training binary classifier to distinguish **Benign vs Malignant tumors**

---

### 🔹 2. Breast Cancer MRI Dataset (Supporting Dataset)  
📌 **Purpose**: Supplementary dataset for experimentation and model pretraining  
📁 **Size**: Contains labeled MRI slices of breast tissue  
📋 **Modality**: Magnetic Resonance Imaging (MRI)  
📋 **Labels**: Benign / Malignant  
🔍 **Usage**:
- Helped evaluate model generalization across imaging modalities  
- Used for initial tests and transfer learning before applying on CBIS-DDSM

> 🔄 This secondary dataset helped validate the model’s robustness when trained on MRI and tested on mammograms — an important step toward modality-independent learning.

---

## ⚙️ Project Workflow

<details>
<summary><strong>1. Data Preprocessing</strong></summary>

- Images extracted from DICOM or PNG.
- Resized to 224×224 for CNN input.
- Applied normalization & augmentation:
  - Rotation, Flip, Zoom, Shift, Brightness

</details>

<details>
<summary><strong>2. Model Architecture</strong></summary>

- Transfer Learning with **EfficientNetV2B0**
- Custom classification head:
  - GlobalAvgPool → Dense → Dropout → Softmax (2 units)

</details>

<details>
<summary><strong>3. Training Configuration</strong></summary>

- Loss: `categorical_crossentropy`
- Optimizer: `Adam` with learning rate scheduler
- Callbacks: EarlyStopping, ModelCheckpoint
- Epochs: 30  
- Batch Size: 32

</details>

<details>
<summary><strong>4. Evaluation</strong></summary>

- Accuracy  
- Precision, Recall, F1-Score  
- AUC-ROC Curve  
- Confusion Matrix

</details>

<details>
<summary><strong>5. Explainability</strong></summary>

- **Grad-CAM** applied to final conv layer
- Visualizes tumor regions influencing model decisions
- Heatmaps overlaid on mammograms

</details>

---

## 📊 Results

| Metric        | Value (%) |
|---------------|-----------|
| Accuracy      | 93.7      |
| Precision     | 92.5      |
| Recall        | 94.1      |
| F1-Score      | 93.3      |
| AUC-ROC       | 0.961     |

---

## 🧰 How to Use

### 🔸 Clone this repository

```bash
git clone https://github.com/yourusername/breast-cancer-detection-dl.git
cd breast-cancer-detection-dl
```

### 🔸 Setup your environment

```bash
pip install -r requirements.txt
```

### 🔸 Prepare the dataset

- Download from:
  - [CBIS-DDSM Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- Place image folders inside:  
  `data/CBIS_DDSM/` and `data/Second_Dataset/` respectively.

### 🔸 Run on Google Colab

Click badge 👉 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19Jhi6cM7LTknwCYVK1xU9V9poDDOKRuz)

---

## 🚀 Future Work

- 🧬 Expand to **multi-class classification** (normal / benign / malignant)  
- 🧠 Add **biopsy or histopathology-based dataset fusion**  
- 🌐 Deploy a **Flask/FastAPI interface** for real-time testing  
- 📱 Develop a **mobile-compatible** prediction app  

---

## 🔗 References

1. **Lee, Roger S., et al.** _"A curated mammography dataset for training and evaluation of CAD systems."_ Scientific Data, 2017.  
2. **CBIS-DDSM Dataset** – [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)  
3. **EfficientNetV2** – [Google AI Blog](https://ai.googleblog.com/2021/08/efficientnetv2-smaller-models-and.html)  
4. **Grad-CAM** – Selvaraju et al., _“Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”_

---

## 🪪 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

## 🙋‍♂️ Author & Contact

**Abhinav Jajoo**  
📧 abhinav@example.com  
📄 [LinkedIn Profile](https://linkedin.com/in/your-profile)

---

> "AI won’t replace radiologists, but radiologists who use AI will replace those who don’t." – Prof. Keith Dreyer, Harvard Medical School
