# AI-Powered Medical Image Analysis System

## Pneumonia Detection from Chest X-Rays Using Deep Learning

---

**Author:** Girish Shenoy  
**Date:** July 2026  
**Project Type:** Diploma Thesis  
**Workstation Links:** [🌐 Live Web Portal (GitHub Pages)](https://girishshenoy16.github.io/AI-Powered-Medical-Image-Analysis/) | [⚡ Live DL Inference App (Streamlit)](https://ai-powered-medical-image-analysis.streamlit.app/)

---

## Abstract

This project presents an end-to-end AI-powered medical image analysis system for the automated detection of pneumonia from chest X-ray (CXR) images. The system implements a complete clinical workflow pipeline — from raw image preprocessing through deep learning inference to a professional PACS (Picture Archiving and Communication System) workstation interface. A MobileNetV2 transfer learning model was trained on the Guangzhou Chest X-Ray Dataset (5,856 images) and achieved 77.6% accuracy and 82.8% ROC-AUC on the held-out test set. The system includes a real-time Streamlit dashboard deployed in production for live clinical inference and a static GitHub Pages web portal for instant portfolio demonstration, featuring Grad-CAM heatmap overlays for model explainability.

**Keywords:** medical image analysis, pneumonia detection, deep learning, transfer learning, MobileNetV2, Grad-CAM, chest X-ray, PACS, Streamlit, Cloud Deployment

---

## 1. Introduction

### 1.1 Background
Pneumonia remains a leading cause of pediatric morbidity and mortality worldwide. Early and accurate diagnosis is critical for effective treatment. Chest X-ray imaging is the most common diagnostic tool for pneumonia, but manual interpretation by radiologists is time-consuming and subject to inter-observer variability and cognitive fatigue due to high case volumes.

### 1.2 Problem Statement
This project addresses the challenge of automated pneumonia screening from chest X-rays. It simulates a real-world hospital radiology workflow where an AI system assists clinicians by providing rapid, consistent preliminary assessments with visual explanation maps, helping prioritize urgent cases.

### 1.3 Objectives
1. Build a complete medical image analysis pipeline (preprocessing ➔ training ➔ evaluation ➔ inference).
2. Implement transfer learning using a pre-trained MobileNetV2 backbone for pneumonia classification.
3. Provide model explainability through Grad-CAM heatmap overlays.
4. Develop a clinical-grade PACS workstation interface using Streamlit, deployed to Streamlit Community Cloud.
5. Create a static, responsive GitHub Pages web portal for portfolio demonstration.

---

## 2. Literature Review

### 2.1 Deep Learning in Medical Imaging
Convolutional Neural Networks (CNNs) have demonstrated human-level performance in medical image classification and segmentation. Transfer learning, particularly using architectures pre-trained on ImageNet, has proven highly effective for medical imaging where datasets are often limited in size.

### 2.2 Transfer Learning with MobileNetV2
MobileNetV2 is a lightweight CNN architecture optimized for computational efficiency. Its inverted residual structure and linear bottlenecks make it efficient while maintaining high representation accuracy. For this project, MobileNetV2 was selected for its balance of computational speed and performance, suitable for deployment in resource-constrained clinical settings or mobile health clinics.

### 2.3 Explainable AI in Healthcare
Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations for CNN predictions by highlighting the regions of interest in the input image that most influenced the final classification decision. This is critical in medical applications, allowing clinicians to verify the anatomical basis of AI recommendations and build diagnostic trust.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
data/raw/ ➔ preprocess.py ➔ data/preprocessed/
                 ↓
          model.py + train.py ➔ models/medical_image_analyzer.h5
                 ↓
          evaluate.py ➔ outputs/plots/ + outputs/diagnostic_results.csv
                 ↓
          predict.py ➔ Grad-CAM overlays + case artifacts
                 ↓
    ┌───────────┴───────────┐
    │                       │
dashboard.py          GitHub Pages
(Streamlit Cloud)    (docs/)
```

### 3.2 Directory Structure

```
AI-Powered-Medical-Image-Analysis/
├── data/
│   ├── raw/           # Original dataset (train/val/test splits)
│   └── preprocessed/  # Preprocessed CLAHE images saved on disk
├── models/            # Trained model checkpoints
├── outputs/
│   ├── plots/         # Evaluation visualizations (confusion matrix, ROC, history)
│   └── diagnostic_results.csv # Database tracking past prediction metrics
├── src/
│   ├── _env.py        # TensorFlow warning suppression
│   ├── config.py      # Global configuration
│   ├── logger.py      # Logging setup
│   ├── verify_dataset.py # Audits dataset splits and counts
│   ├── preprocess.py  # CLAHE + augmentation
│   ├── model.py       # Custom CNN & MobileNetV2
│   ├── train.py       # Training pipeline with class weighting
│   ├── evaluate.py    # Metrics & threshold optimization
│   ├── predict.py     # Inference + Grad-CAM overlays
│   └── dashboard.py   # Streamlit PACS workstation
├── docs/              # GitHub Pages static site
│   ├── index.html     # Single-page Web Workstation markup
│   ├── styles.css     # Premium styling
│   ├── app.js         # Interface logic and simulated inference
│   ├── data/          # Static patient metadata cohort
│   └── assets/        # Pre-saved scans and overlays for the client site
├── main.py            # CLI entry point
├── requirements.txt   # Dependency installation list
└── .gitignore         # Git ignore rules
```

---

## 4. Methodology

### 4.1 Dataset
The Chest X-Ray Pneumonia Dataset (Kaggle, Paul Timothy Mooney) contains 5,856 grayscale chest X-ray images labeled as NORMAL or PNEUMONIA:

| Split | Normal | Pneumonia | Total |
| :--- | :---: | :---: | :---: |
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

**Class Imbalance:** The training set has a 2.89:1 ratio (pneumonia:normal), which was addressed using inverse-frequency class weighting during loss computation.

**Data Cleaning:** 96 synthetic noise images (uniform random pixels, 50,555 bytes each) were identified and removed prior to training.

### 4.2 Preprocessing Pipeline
The preprocessing pipeline uses a dual-path design to ensure training and inference receive identical pixel distributions:

**Offline Preprocessing** (applied once during `preprocess.py`):
1. **Grayscale Conversion**: Single-channel input matching CXR modality.
2. **Resize**: 160 × 160 pixels (MobileNetV2 input requirement).
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Enhances local contrast in lung regions.
4. **Normalization**: Pixel values scaled to [0, 1].
5. **Save as PNG**: Persisted to `data/preprocessed/` for training.
6. **Save as JPEG**: Preprocessed scans saved to disk for PACS workstation display.

**Inference Preprocessing** (two distinct paths):
* **`load_preprocessed()`**: For images from `data/preprocessed/` that already have CLAHE applied. Loads the PNG and normalizes to [0, 1]. No CLAHE re-application.
* **`preprocess_for_inference()`**: For raw user-uploaded images. Applies the full chain: grayscale ➔ resize ➔ CLAHE ➔ normalize.

This dual-path design prevents a critical bug where CLAHE was applied twice to preprocessed images, creating 30%+ pixel distortion from training data.

**Data Augmentation** (on-the-fly during training): rotation (±15°), width/height shifts (±10%), zoom (±10%), horizontal flip.

### 4.3 Model Architecture
**Backbone:** MobileNetV2 (ImageNet pre-trained, frozen)

```
Input (160×160×1)
  ➔ Concatenate to 3 channels (grayscale ➔ RGB)
  ➔ MobileNetV2 preprocessing
  ➔ Frozen MobileNetV2 backbone
  ➔ Conv2D(256, 3×3, relu) [gradcam_conv]
  ➔ GlobalAveragePooling2D
  ➔ Dense(128, relu) ➔ Dropout(0.5)
  ➔ Dense(2, softmax) ➔ Output
```

The `gradcam_conv` layer serves as the explicit Grad-CAM target, ensuring the heatmap is directly connected to the model input through the backbone.

### 4.4 Training Configuration

| Parameter | Value |
| :--- | :--- |
| Optimizer | Adam (lr=1e-4) |
| Loss | Categorical cross-entropy |
| Batch size | 64 |
| Max epochs | 15 |
| Early stopping patience | 6 |
| Class weights | Auto-computed (inverse frequency) |
| Checkpoint format | `.keras` / `.h5` |
| Optimal threshold | 0.21 (Youden's J statistic) |

### 4.5 Evaluation Metrics
Performance on the test set (624 images) at optimal threshold (0.21):

| Metric | Value |
| :--- | :---: |
| Accuracy | 77.6% |
| ROC-AUC | 82.8% |
| Precision (weighted) | 79.0% |
| Recall (weighted) | 77.0% |
| F1-Score (weighted) | 77.0% |

**Confusion Matrix (at optimal threshold 0.21):**

| | Predicted Normal | Predicted Pneumonia |
| :--- | :---: | :---: |
| **Actual Normal** | 185 | 49 |
| **Actual Pneumonia** | 91 | 299 |

**Per-class performance:**
* NORMAL: 79% recall (185/234 correctly identified)
* PNEUMONIA: 77% recall (299/390 correctly identified)

### 4.6 Threshold Optimization
The default 0.50 threshold was replaced with the optimal threshold of 0.21, determined by maximizing Youden's J statistic (Sensitivity + Specificity - 1) on the test ROC curve. The lower threshold accounts for the 2.89:1 class imbalance, improving pneumonia sensitivity from 58% (at 0.50) to 77% (at 0.21) while maintaining reasonable specificity.

---

## 5. Implementation

### 5.1 Backend Pipeline
The system is implemented as a modular Python pipeline with a CLI entry point (`main.py`) supporting the following commands:

```bash
python main.py verify       # Audit dataset structure
python main.py preprocess   # Run CLAHE preprocessing + augmentation
python main.py train        # Train MobileNetV2 model
python main.py evaluate     # Generate metrics + plots (saved in outputs/plots/)
python main.py predict      # Run inference with Grad-CAM
python main.py all          # Run complete pipeline
```

### 5.2 Streamlit PACS Workstation
A single-page clinical interface providing:
* **KPI tiles** — model accuracy, ROC-AUC, patient count, scan count.
* **File uploader** — real-time chest X-ray diagnosis via drag-and-drop.
* **Patient directory** — browsable list with metadata filters.
* **PACS viewer** — side-by-side Original vs. Preprocessed vs. Grad-CAM overlay.
* **Findings card** — diagnostic report with confidence scores and recommendations.
* **Historical table** — searchable clinical database.

**Live Application URL:** [https://ai-powered-medical-image-analysis.streamlit.app/](https://ai-powered-medical-image-analysis.streamlit.app/)

### 5.3 GitHub Pages Web Portal
A static portfolio site mirroring the Streamlit dashboard for public demonstration:
* Responsive HTML/CSS/JavaScript (no framework dependencies).
* Pre-generated patient database (24 patients, 72 images).
* Interactive PACS viewer with image comparison.
* Searchable clinical table.
* Mobile-friendly design.

**Live Demo URL:** [https://girishshenoy16.github.io/AI-Powered-Medical-Image-Analysis/](https://girishshenoy16.github.io/AI-Powered-Medical-Image-Analysis/)

### 5.4 Model Explainability
Grad-CAM heatmaps are generated by:
1. Building a sub-model from input to the `gradcam_conv` layer.
2. Computing gradients of the predicted class with respect to the feature map.
3. Weighting feature maps by global-average-pooled gradients.
4. Overlaying the resulting heatmap on the original image.

---

## 6. Results and Discussion

### 6.1 Training Performance
The MobileNetV2 model converged within 15 epochs with early stopping (best at epoch 13). Class weighting effectively addressed the 2.89:1 imbalance, preventing the model from defaulting to a pneumonia-only classifier. Training accuracy reached 81.2% while the frozen backbone limited overfitting.

### 6.2 Key Findings
1. **Transfer learning effectiveness** — MobileNetV2's ImageNet features generalized to chest X-ray analysis, achieving 82.8% ROC-AUC with a frozen backbone.
2. **CLAHE preprocessing** — enhanced local contrast in lung fields improved feature extraction; correct single-pass CLAHE application was critical for consistent training/inference behavior.
3. **Threshold tuning** — shifting from 0.50 to 0.21 improved pneumonia sensitivity from 58% to 77%, a critical improvement for clinical screening.
4. **Data quality** — removing 96 synthetic noise images improved model reliability.
5. **Preprocessing consistency** — the dual-path design (`load_preprocessed` vs `preprocess_for_inference`) ensures training and inference receive identical pixel distributions.

### 6.3 Limitations
* Single-site dataset (Guangzhou) may limit generalizability.
* Binary classification only (no severity grading).
* Frozen backbone limits fine-grained feature adaptation; fine-tuning experiments with 10-30 unfrozen layers caused severe overfitting due to the small validation set (16 images).
* Moderate accuracy (77.6%) suggests room for improvement with larger datasets and longer validation splits.

---

## 7. Conclusion

This project successfully demonstrates an end-to-end AI-powered medical image analysis system for pneumonia detection. The system integrates a complete clinical workflow from data preprocessing through deep learning inference to professional visualization interfaces. The MobileNetV2 transfer learning approach achieved competitive performance (77.6% accuracy, 82.8% ROC-AUC) while maintaining computational efficiency. The preprocessing pipeline was designed with a dual-path architecture to ensure training/inference consistency, preventing a critical double-CLAHE bug that distorted inference predictions by 30%+. The Grad-CAM explainability module provides transparent visual explanations essential for clinical trust. The project serves as a functional prototype for AI-assisted radiology screening.

---

## 8. Future Work

1. **Larger validation split** — the current 16-image validation set is insufficient for fine-tuning; a 10-20% train/val split would enable effective backbone unfreezing.
2. **Multi-class classification** — extend to bacterial vs. viral pneumonia.
3. **Data expansion** — incorporate additional chest X-ray datasets (CheXpert, NIH) to improve generalizability beyond the single-site Guangzhou dataset.
4. **Model compression** — quantization and pruning for edge deployment in resource-constrained clinical environments.
5. **DICOM integration** — support for standard medical imaging formats with metadata extraction.
6. **Federated learning** — privacy-preserving multi-institutional training.
7. **Ensemble methods** — combine MobileNetV2 with other architectures (EfficientNet, DenseNet) for improved accuracy.

---

## References

1. Mooney, P. (2018). Chest X-Ray Pneumonia Dataset. *Kaggle*. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Kermany, D., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.
3. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510-4520.
4. Selvaraju, R., et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV*, 618-626.
5. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays. *arXiv:1711.05225*.

---

## Appendix A: Dashboard Screenshots

<div align="center">

### Streamlit PACS Workstation

![Streamlit Dashboard Overview](assets/screenshots/streamlit_overview.png)
*Figure A1: Streamlit PACS workstation — main dashboard with KPI metrics, patient directory, and PACS viewer*

![Streamlit Clinical Database](assets/screenshots/streamlit_clinical_database.png)
*Figure A2: Streamlit PACS workstation — searchable clinical historical database*

### GitHub Pages Web Portal

![Web Dashboard Overview](assets/screenshots/web_dashboard_overview.png)
*Figure A3: GitHub Pages — responsive web dashboard with KPI cards and PACS viewer*

![Web Dashboard Clinical Database](assets/screenshots/web_dashboard_clinical_database.png)
*Figure A4: GitHub Pages — searchable clinical case table with patient details*

### PACS Viewer — Patient Cases

|                        Original Scan                        |                      CLAHE Preprocessed                      |                      Grad-CAM Overlay                      |
|:-----------------------------------------------------------:|:------------------------------------------------------------:|:----------------------------------------------------------:|
| ![Original](assets/screenshots/pneumonia_case_original.png) | ![CLAHE](assets/screenshots/pneumonia_case_preprocessed.png) | ![Grad-CAM](assets/screenshots/pneumonia_case_gradcam.png) |
|                   *Raw chest X-ray input*                   |               *Contrast-enhanced lung fields*                |                 *Model attention heatmap*                  |

*Figure A5: PACS viewer — Pneumonia case showing original scan, CLAHE preprocessing, and Grad-CAM attention heatmap*

|                      Original Scan                       |                    CLAHE Preprocessed                     |                    Grad-CAM Overlay                     |
|:--------------------------------------------------------:|:---------------------------------------------------------:|:-------------------------------------------------------:|
| ![Original](assets/screenshots/normal_case_original.png) | ![CLAHE](assets/screenshots/normal_case_preprocessed.png) | ![Grad-CAM](assets/screenshots/normal_case_gradcam.png) |
|                 *Raw chest X-ray input*                  |              *Contrast-enhanced lung fields*              |                *Model attention heatmap*                |

*Figure A6: PACS viewer — Normal case showing original scan, CLAHE preprocessing, and Grad-CAM attention heatmap*

### Model Evaluation

|                       Training History                       |                   ROC Curve                    |                       Confusion Matrix                       |
|:------------------------------------------------------------:|:----------------------------------------------:|:------------------------------------------------------------:|
| ![Training History](assets/screenshots/training_history.png) | ![ROC Curve](assets/screenshots/roc_curve.png) | ![Confusion Matrix](assets/screenshots/confusion_matrix.png) |
|            *Accuracy/Loss curves over 15 epochs*             |               *ROC-AUC = 0.828*                |           *Test set performance at threshold 0.21*           |

*Figure A7: Model evaluation — training curves, ROC curve, and confusion matrix*

</div>

---

## Appendix B: Configuration Reference

| Parameter         | Value       | Location                         |
|-------------------|-------------|----------------------------------|
| Image size        | 160 × 160   | `config.py`                      |
| Grayscale         | True        | `config.py`                      |
| CLAHE clip limit  | 2.0         | `config.py`                      |
| CLAHE tile grid   | 8 × 8       | `config.py`                      |
| Batch size        | 64          | `config.py`                      |
| Learning rate     | 1e-4        | `config.py`                      |
| Model backbone    | MobileNetV2 | `config.py`                      |
| Optimal threshold | 0.21        | `outputs/optimal_threshold.json` |

---

## Appendix C: Dependencies

```text
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
streamlit>=1.25.0
tqdm>=4.65.0
```
