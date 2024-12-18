---

# **Automated Detection and Classification of Diabetic Retinopathy Using Deep Learning**

This repository contains the implementation of a deep learning-based approach to automate the detection and classification of diabetic retinopathy (DR) from retinal fundus images. The project leverages advanced segmentation, classification models, and Explainable AI (XAI) techniques for accurate and transparent diagnosis.

---

## **Introduction**  
Diabetic retinopathy (DR) is a leading cause of blindness among individuals with diabetes. Early detection and diagnosis are essential to prevent vision loss and manage the disease effectively.  

Traditional DR diagnosis relies on manual examination of fundus images, which can be slow and error-prone. This project employs deep learning techniques to automate and enhance the accuracy of DR detection and classification, enabling faster and more reliable disease management.  

---

## **Problem Statement**  
The project addresses the following challenges:  
- **Image Variability:** Fundus images vary significantly in quality, lighting, and resolution.  
- **Lesion Detection:** Small and variable lesions, such as microaneurysms, hemorrhages, and exudates, are difficult to detect and segment.  
- **Classification:** Combining lesion detection with robust classification into different DR severity levels for comprehensive diagnostics.  
- **Explainable AI:** Integrating methods like LIME and Grad-CAM to provide insights into model predictions, aiding clinical interpretation.  

---

## **Datasets**  
1. **Segmentation:** IDRiD Dataset (Indian Diabetic Retinopathy Image Dataset) with annotations for:  
   - Microaneurysms (MA)  
   - Hemorrhages (HE)  
   - Exudates (EX)  
   - Soft Exudates (SE)  
   - Optic Disc (OD)  
2. **Classification:** APTOS 2019 Dataset for DR severity grading.  

---

## **Pipeline Overview**  
1. **Image Preprocessing:**  
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)  
   - Color enhancement  
   - Normalization and resizing (512x512)  
2. **Data Augmentation:**  
   - Random rotations, cropping, brightness, and color adjustments for robustness.  
3. **Model Training:**  
   - **Segmentation Models:**  
     - UNet  
     - GCN with ResNet backbone  
     - UNet with VGG16 backbone  
   - **Classification Models:**  
     - Custom CNN  
     - ResNet-34  
     - Vision Transformer (ViT) with Grad-CAM for XAI.  
4. **Evaluation Metrics:**  
   - Segmentation: Dice Loss, BCE Loss, IoU  
   - Classification: Test Accuracy, AUC  

---

## **Results**

### **Segmentation**
- **UNet + CLAHE (Green Channel):**  
  - **AUC:** 91%  
  - **AUPR:** 55%  
  - Enhanced detection of retinal abnormalities using the green channel.  
- **UNet + VGG16 Backbone:**  
  - **Dice Score:** 94.96%  
  - **IoU:** 91%  

### **Classification**
- **Custom CNN:**  
  - Moderate accuracy, effective for mild and moderate cases.  
  - **Test Accuracy:** 71.67%  
- **ResNet-34:**  
  - Strong performance across all severity levels.  
  - **Test Accuracy:** 84.02%  
- **Vision Transformer (ViT):**  
  - **Test Accuracy:** 87.62%  
  - **Test AUC:** 91.05%  

---

## **Explainable AI**
- Integrated Grad-CAM (XAI) to provide visual explanations and highlight critical regions in fundus images, improving model interpretability for clinicians.  

---

## **Future Work**  
- Enhance segmentation models with advanced architectures like DeepLab.  
- Optimize classification models for real-time predictions.  
- Integrate additional XAI techniques for better clinical explainability.  

---

