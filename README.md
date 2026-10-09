# RetinaAI: diabetic retinopathy segmentation and grading

**Find the lesions, then grade the eye.** RetinaAI is a set of notebooks that segment diabetic retinopathy (DR) lesions in retinal fundus images and classify DR severity into five grades. It compares U-Net variants for segmentation and a dense baseline against a fine-tuned ResNet-34 for grading.

> B.Tech Artificial Intelligence & Data Science major project (Phase I), Shiv Nadar University Chennai, November 2024. Supervisor: Ms Dhivya S.

## The problem

DR damages the blood vessels of the retina. It is a leading cause of blindness in people with diabetes, and early detection prevents much of that vision loss.

- **Manual grading is slow.** An ophthalmologist reads each fundus image by eye, which is slow and error-prone in large screening programmes.
- **Lesions are tiny.** Microaneurysms, haemorrhages and exudates cover a very small fraction of the image, so the classes are heavily imbalanced against background.
- **Images vary.** Quality, lighting and resolution differ from camera to camera.
- **Clinicians need to see why.** A grade without an explanation is hard to trust, so the project also looks at explainable AI (XAI).

## Approach

| Stage | What the notebooks do |
|---|---|
| **Lesion segmentation (U-Net)** | PyTorch U-Net and a GCN (ResNet-152 encoder) on IDRiD lesion masks. CLAHE, green-channel-only input, rotations and colour jitter. Losses compared: MSE, BCE, Dice and mean Dice. Metrics: ROC AUC and AUPR. |
| **Lesion segmentation (U-Net + VGG16)** | Keras U-Net with an ImageNet VGG16 encoder on IDRiD + DIARETDB1 at 1024 × 1024. Three classes: background, red lesions (MA + HE) and exudates (EX + SE). Categorical focal loss, 5-fold cross-validation, Dice and IoU. |
| **Severity grading (baseline)** | Keras dense network on Gaussian-blurred 224 × 224 APTOS 2019 images, five grades (No DR, Mild, Moderate, Severe, Proliferative DR). |
| **Severity grading (ResNet-34)** | fastai ResNet-34 pre-trained on ImageNet, trained frozen, then unfrozen and fine-tuned with augmentation. |
| **Early field study** | K-means clustering, SGD, logistic regression, a small CNN and an ANN on 516 images, before and after Canny + k-means segmentation. |
| **Explainability** | Vision Transformer with Grad-CAM, described in the project report. The code for this part is not in the repository. |

## How it works

```
                  fundus image
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
  SEGMENTATION                   CLASSIFICATION
  IDRiD (+ DIARETDB1)            APTOS 2019, 224×224
        │                              │
  resize · crop · pad            Gaussian blur / fastai aug_transforms
  CLAHE · green channel          ImageNet normalisation
  rotate · colour jitter               │
        │                        ┌─────┴──────┐
  ┌─────┴───────────┐            ▼            ▼
  ▼                 ▼        dense net    ResNet-34
  U-Net / GCN   U-Net + VGG16   (Keras)    (fastai, fine-tuned)
  (PyTorch)     (Keras, 5-fold)    │            │
  │                 │              └─────┬──────┘
  AUC · AUPR    Dice · IoU               ▼
  per lesion    3 classes        5 grades: No DR … Proliferative
```

**Notable design decisions:**
- **Green channel only.** In the U-Net runs, green-only models beat full RGB. The report puts this down to the green channel's higher sensitivity to retinal structures.
- **Imbalance-aware losses.** Dice, focal and categorical focal loss were tried because lesion pixels are rare compared with background.
- **Model checkpoints on the metric that matters.** The PyTorch U-Net saves the model with the best validation AUPR. The VGG16 U-Net saves on best validation Dice and stops early.
- **Transfer learning for grading.** ResNet-34 starts from ImageNet weights. The report uses the same idea for the ViT.

## Datasets

None of the data is included in the repository. The notebooks expect these datasets at the paths shown.

| Dataset | Used for | Path in the notebook |
|---|---|---|
| IDRiD segmentation set (54 train / 27 test images; MA, HE, EX, SE and OD masks) | `Segmentation_U-Net.ipynb` | `drive/My Drive/Segmentation.nosync/` with `train_images/`, `train_masks/<task>/`, `test_images/`, `test_masks/` |
| IDRiD + DIARETDB1 | `Segmentation_U-Net_VGG_Backbone.ipynb` | `/content/gdrive/MyDrive/Colab Notebooks/DR_data/` with `image/` and `mask/{MA,HE,EX,SE}/` |
| APTOS 2019, resized to 224 × 224 (Kaggle dataset `diabetic-retinopathy-224x224-2019-data`) | `Classification_Resnet34.ipynb` | `/kaggle/input/diabetic-retinopathy-224x224-2019-data` |
| 516 labelled fundus images | `Segmentation_And_Classification_of_Diabetic_Retinopathy_FieldStudy.ipynb` | hard-coded local Windows paths |

## Results

All numbers below come from saved notebook outputs or from the project report (`Project_Report.pdf`). They come from different datasets and splits, so they are not directly comparable.

### Segmentation

| Model | Setting | Result | Source |
|---|---|---|---|
| U-Net, MSE loss, green channel | CLAHE only, exudate (EX) masks | **AUC 91%, AUPR 55%** | Results cell in `Segmentation_U-Net.ipynb`; report §3.7 |
| U-Net, MSE loss, green channel | No image enhancer | AUC 90%, AUPR 56% | Same results cell; report Table 3.1 |
| U-Net, MSE loss, green channel | Colour jitter + CLAHE | AUC 87%, AUPR 56% | Same results cell; report Table 3.1 |
| U-Net, MSE loss, RGB | CLAHE only | AUPR 46% | Same results cell; report Table 3.1 |
| U-Net + VGG16, categorical focal loss | 5-fold cross-validation | mean Dice **90.44%**, mean IoU **84.02%** | Output of `Segmentation_U-Net_VGG_Backbone.ipynb` |
| U-Net + VGG16, categorical focal loss | Held-out test set (36 images), best fold (k = 2) | Dice **94.96%**, IoU **91.00%** | Output of `Segmentation_U-Net_VGG_Backbone.ipynb`; report §3.7 |
| U-Net + VGG16, categorical focal loss | — | AUPR 68% (green) / 53% (RGB), AUC 93% | Report Table 3.1 only (not in the notebook output) |

The results cell in `Segmentation_U-Net.ipynb` lists 32 loss and preprocessing combinations. The best AUPR was 56%, from MSE loss on the green channel. The saved training run in that notebook ends with a best mean validation AUPR of 0.5263.

The VGG16 Dice and IoU are averaged over all three one-hot channels, including background, so they read higher than lesion-only scores would.

### Severity grading (5 classes)

| Model | Split | Accuracy | Cohen's κ | MCC | Source |
|---|---|---|---|---|---|
| Dense network baseline (Keras) | 2,563 train / 769 val / 330 test | **65.15%** test (67.49% val) | 0.399 | 0.435 | Output of `Classification_Resnet34.ipynb` ("DR V1") |
| ResNet-34 (fastai), after fine-tuning | 2,930 train / 732 validation (20% hold-out) | **84.02%** validation | 0.802 | 0.803 | Output of `Classification_Resnet34.ipynb` ("DR V2") |
| Vision Transformer + Grad-CAM | — | **87.62%** test, AUC 91.05% | — | — | Report §4.4.1 only (code not in the repo) |

ResNet-34 per-class F1 on the validation set (notebook output):

| No DR | Mild | Moderate | Severe | Proliferative DR |
|---|---|---|---|---|
| 0.99 | 0.69 | 0.82 | 0.53 | 0.69 |

The baseline almost never predicts Severe or Proliferative DR. Its F1 is 0 for both classes. ResNet-34 is much better but still weakest on Severe.

### Early field study (516 images)

| Model | Original images | After Canny + k-means segmentation |
|---|---|---|
| K-means (k = 4), clusters mapped to labels | 38.4% | — |
| SGD classifier | 34.2% | 37.0% |
| Logistic regression | 53.4% | 38.4% |

Test-set accuracy on 73 images, from the notebook outputs.

## Repository layout

```
├── Segmentation_U-Net.ipynb                        PyTorch U-Net and GCN on IDRiD; CLAHE, green channel, AUC/AUPR, TensorBoard
├── Segmentation_U-Net_VGG_Backbone.ipynb           Keras U-Net with VGG16 encoder; IDRiD + DIARETDB1; focal loss; 5-fold CV
├── Classification_Resnet34.ipynb                   Dense-network baseline (Keras) and ResNet-34 (fastai) on APTOS 2019
├── Segmentation_And_Classification_of_Diabetic_Retinopathy_FieldStudy.ipynb
│                                                   Early study: k-means, Canny segmentation, SGD, logistic regression, CNN, ANN
├── Project_Report.pdf                              Phase I project report (methods, results, ViT + Grad-CAM)
└── Investigations of Explainable AI based Solutions for Diabetic Retinopathy Diagnosis.pdf
                                                    Team review paper on XAI methods (LIME, SHAP, Grad-CAM) for DR
```

## Running the notebooks

There is no `requirements.txt`. Each notebook was run on a hosted GPU and installs or imports what it needs.

| Notebook | Ran on | Main libraries |
|---|---|---|
| `Segmentation_U-Net.ipynb` | Google Colab (Python 3.6, GPU) | PyTorch, torchvision, SimpleITK, albumentations 0.1.x, scikit-learn, TensorBoard |
| `Segmentation_U-Net_VGG_Backbone.ipynb` | Google Colab with Google Drive mounted | TensorFlow / Keras, OpenCV |
| `Classification_Resnet34.ipynb` | Kaggle (2 × Tesla T4) | TensorFlow / Keras, fastai, PyTorch, scikit-learn, OpenCV |
| `...FieldStudy.ipynb` | Local Windows machine | scikit-learn, OpenCV, PyTorch |

1. Download the dataset for the notebook you want (see [Datasets](#datasets)).
2. Upload it to Colab/Drive or attach it to a Kaggle notebook, or edit the path variables at the top of the notebook.
3. Run the cells in order. The segmentation notebooks save checkpoints to Google Drive or `/content/save/`.

## Limitations

- **Datasets and trained weights are not included.** Paths are hard-coded for Colab, Kaggle or a local Windows machine.
- **ViT and Grad-CAM code is missing.** The ViT accuracy and the XAI part appear only in the report.
- **The field-study CNN accuracy is not valid.** `validation_epoch_end` adds a constant `+0.45` to the validation accuracy, so its printed `val_acc` (0.65 and 0.7167) is inflated. The value without the offset is 0.27. This README does not report that CNN.
- **ResNet-34 accuracy is validation accuracy.** It is measured on the 20% hold-out used during fine-tuning, not on a separate test set.
- **No shared test set.** The segmentation and grading experiments use different datasets and splits, so the numbers can't be compared across models.
- **Old library versions.** The U-Net notebook uses the albumentations 0.1.x import path and Python 3.6. fastai warns that `cnn_learner` is deprecated.
- **Not a medical device.** This is research code and must not be used for diagnosis.

## Team

- Lohesh M
- P Esvar Ram Kumar
- Rysha M Rafeeque
- P Rathna Sabapathy

Supervisor: Ms Dhivya S, Assistant Professor. Head of Department: Dr T. Nagarajan. Department of Computer Science and Engineering, Shiv Nadar University Chennai.

The review paper *Investigations of Explainable AI based Solutions for Diabetic Retinopathy Diagnosis* is by Rysha M Rafeeque, Lohesh M, Rathna Sabapathy P, Esvar Ram Kumar P and Dhivya S.

## Licence

MIT. See [LICENSE](LICENSE). This covers the code in this repository.

- **Datasets.** IDRiD, DIARETDB1 and APTOS 2019 are third-party datasets with their own terms. The IDRiD folder used here ships a CC-BY-4.0 licence file.
- **Borrowed code.** Parts of the GCN in `Segmentation_U-Net.ipynb` are marked in the code as borrowed from [ycszen/pytorch-ss](https://github.com/ycszen/pytorch-ss/blob/master/gcn.py). The focal loss follows [Lin et al.](https://arxiv.org/pdf/1708.02002.pdf).
- **Pre-trained weights.** VGG16, ResNet-34 and ResNet-152 come from Keras and torchvision.
