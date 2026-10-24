# Multimodal-Glaucoma-Detection-from-OCT-Volumes-and-RNFLT-Maps
This project focuses on AI-based medical image analysis, aiming to develop an interpretable multimodal model for glaucoma detection using Optical Coherence Tomography (OCT) images and Retinal Nerve Fiber Layer Thickness (RNFLT) maps, while also estimating the uncertainty of its predictions.

> **Research use only.** The model is not a clinically validated diagnostic model. Its outputs should not be used for clinical decisions without appropriate independent validation.

## Project at a glance

- **Learn from unlabeled OCT:** train a DINO teacher–student encoder using OCT B-scans without diagnostic labels.
- **Combine complementary structural inputs:** fuse features from 12 depth-sampled OCT B-scans with features from a single-channel retinal nerve fiber layer thickness (RNFLT) map.
- **Evaluate beyond the development test set:** train and select the model using Harvard-GF, then evaluate the locked model and operating on held-out Harvard-GDP subjects.
- **Study prediction uncertainty:** use MC dropout to assess prediction uncertainty and whether prediction variability helps identify classification errors.
- **model interpretability:** use Grad-CAM, Integrated Gradients, and SSL representation visualizations as interpretability tools.

## Method overview

### 1. DINO self-supervised pretraining for OCT

The OCT encoder is pretrained with a DINO-style teacher–student framework. Pretraining uses **2,600 unlabeled OCT volumes**: the 2,100 subjects in the Harvard-GF training split and 500 additional, Harvard-GDP subjects. Sampling 12 B-scans from each volume produces **31,200 B-scans** for representation learning. Harvard-GF validation and test volumes are not used for SSL pretraining, and diagnostic labels are not part of the SSL objective.

Each B-scan is converted from 8-bit intensity to `[0, 1]`, resized while preserving its aspect ratio, and placed on a `224 × 320` canvas. Two independently augmented views are created for DINO training. The teacher and student use grayscale-adapted ResNet-18 backbones initialized from random weights. A projection head produces 4,096-dimensional self-supervised logits; these logits are used for the DINO objective, not as clinical classes. Training runs for up to 50 epochs with AdamW followed by cosine learning-rate decay, and an EMA teacher. After pretraining, the **teacher ResNet-18 backbone** is retained for downstream classification. 

<img width="975" height="650" alt="image" src="https://github.com/user-attachments/assets/5ec4cbe7-1887-431c-a439-d84b3b1400f2" />

*Figure 1. DINO teacher–student pretraining pipeline.*

### 2. Multimodal OCT + RNFLT glaucoma classifier

The supervised classifier has two branches:

- **OCT branch:** 12 B-scans are sampled along the axial depth dimension at indices `7, 24, 41, 57, 74, 91, 107, 124, 141, 157, 174, 191`. A pretrained, single-channel ResNet-18 extracts a 512-dimensional representation from each slice. A position-aware attention module weights the slice features while retaining information about their relative depth positions, and projects the aggregated OCT representation to 128 dimensions.
- **RNFLT branch:** a separate single-channel residual CNN encodes the RNFLT map and projects it to a 128-dimensional representation. Negative and non-finite RNFLT values are imputed from the nearest valid pixel within the same map. The map is standardized using mean and standard deviation calculated from valid pixels in the Harvard-GF training split, then clipped to `[-5, 5]`.
- **Fusion and prediction:** the two 128-dimensional embeddings are concatenated into a 256-dimensional joint representation and passed through an MLP ending in a single sigmoid glaucoma probability.

During supervised training, mild brightness and gamma augmentation is applied consistently across all 12 OCT slices from a subject; no spatial rotations or flips are used, and the RNFLT map is not augmented.
<img width="1098" height="549" alt="image" src="https://github.com/user-attachments/assets/be92432c-5fb8-40da-a1eb-7ec13b4bb585" />
<img width="1098" height="198" alt="image" src="https://github.com/user-attachments/assets/c6dae4dc-12a2-493e-a326-0d6f93a4fb6d" />

*Figure 2. Two-branch diagnostic model*

The classifier is trained on Harvard-GF with binary cross-entropy and AdamW for up to 25 epochs. The pretrained OCT encoder is initially frozen, and Layer 4 is unfrozen from epoch 4. The checkpoint is selected using validation AUROC alongside validation loss, and early stopping is also applied.

### 3. Sensitivity-constrained operating threshold

Instead of using the default probability threshold of `0.5`, the operating threshold is selected on the **Harvard-GF validation set** to achieve sensitivity of at least 80%, choosing the highest threshold that satisfies the constraint. This procedure yielded a fixed threshold of **`0.404529`**. The threshold is then kept unchanged for both Harvard-GF test evaluation and the held-out Harvard-GDP evaluation. 

The 80% target is a validation-set operating-point criterion, not a guarantee of 80% sensitivity on unseen test data.

## Datasets and evaluation design

### Harvard-GF: supervised development and internal testing

Harvard-GF contains 3,300 subjects with 3D OCT volumes, corresponding 2D RNFLT maps, and binary glaucoma labels. The subject-level split is retained:

| Split | Subjects | Glaucoma positive | Glaucoma negative |
|---|---:|---:|---:|
| Training | 2,100 | 1,083 | 1,017 |
| Validation | 300 | 176 | 124 |
| Test | 900 | 489 | 411 |

Only the training split is used to calculate RNFLT normalization statistics. The validation split is used for checkpoint selection and threshold selection. The internal test split is used for final evaluation.

### Harvard-GDP: held-out cross-dataset evaluation

A separate set of **75 Harvard-GDP subjects** (34 glaucoma-positive and 41 glaucoma-negative) is used for evaluation without supervised retraining, fine-tuning, or threshold recalibration. None of these 75 test subjects overlaps with the 500 GDP subjects used for unlabeled SSL pretraining, and no GDP test samples or labels are used for supervised training, model selection, or threshold selection.

because 500 unlabeled Harvard-GDP volumes were used during SSL pretraining, this is a **held-out cross-dataset evaluation after prior unlabeled exposure to the same data source**, rather than a fully independent external validation with no exposure to the GDP source domain.

## Results

### Internal test: Harvard-GF

Performance on the 900-subject test set at the validation-selected threshold of `0.404529`:

| Metric | Result | 95% confidence interval |
|---|---:|---:|
| Sensitivity | 79.96% | 76.28–83.44% |
| Specificity | 74.94% | 70.56–79.07% |
| Accuracy | 77.67% | 74.89–80.33% |
| Balanced accuracy | 77.44% | 74.54–80.19% |
| Precision | 79.15% | 76.15–82.08% |
| F1-score | 79.55% | 76.95–82.03% |
| AUROC | 85.05% | 82.45–87.37% |
| Average precision | 88.71% | 86.58–90.73% |


<img width="485" height="419" alt="image" src="https://github.com/user-attachments/assets/f32b8e30-6a0e-422f-b01d-14f7a20276ec" />

*Figure 3. Calibration curve for predicted glaucoma probabilities on the internal Harvard-GF test set.* 


### Held-out cross-dataset test: Harvard-GDP

The same Harvard-GF-trained model, RNFLT normalization statistics, and threshold were applied to the 75-subject GDP test set:

| Metric | Result | 95% confidence interval |
|---|---:|---:|
| Sensitivity | 64.71% | 47.06–79.41% |
| Specificity | 92.68% | 85.37–100.00% |
| Accuracy | 80.00% | 70.67–88.00% |
| Balanced accuracy | 78.69% | 69.15–87.27% |
| Precision | 88.00% | 75.86–100.00% |
| F1-score | 74.58% | 60.71–85.71% |
| AUROC | 90.75% | 83.71–96.41% |
| Average precision | 90.19% | 83.09–96.14% |

Confidence intervals were estimated with 2,000 stratified bootstrap replicates.

## Predictive uncertainty and interpretability

### Monte Carlo dropout

For each evaluation sample, **50 stochastic forward passes** are performed. The model remains in evaluation mode while its five existing dropout modules are activated. OCT-encoder and RNFLT-encoder features are computed once per batch and reused across passes; downstream attention, projection, fusion, and classification layers are recomputed with new dropout masks.

The analysis compares MC predictive entropy, mutual information, and predictive variance with deterministic predictive entropy and a distance-to-threshold score. On Harvard-GF, MC predictive variance achieved the highest error-detection AUROC (**0.7258**). On the smaller Harvard-GDP test set, deterministic predictive entropy achieved the highest error-detection AUROC (**0.8044**), closely followed by MC predictive entropy (**0.8000**). These results suggest that uncertainty scores can contain useful information about errors and may help to identify samples at the risk of misclassification. 

### interpretability and explainability 

The project includes:

- **Grad-CAM** on the OCT branch of the multimodal classifier, targeting the final glaucoma logit.
- **Integrated Gradients** for RNFLT input pixels in the preprocessed input space.
- **SSL representation analysis**, including intermediate activation-energy maps and embedding Grad-CAM targeting the norm of the encoder embedding.

<img width="885" height="732" alt="image" src="https://github.com/user-attachments/assets/c984fcf7-8c76-48b9-b4c3-89567389247d" />

*Figure 4. Grad-CAM overlays for selected OCT B-scans and Integrated Gradients attribution maps for the corresponding RNFLT inputs on the internal Harvard-GF test set.*

- <img width="791" height="682" alt="image" src="https://github.com/user-attachments/assets/9da75375-763e-4492-93ec-11559109722c" />
*Figure 5. Grad-CAM overlays for selected OCT B-scans and Integrated Gradients attribution maps for the corresponding RNFLT inputs on the held-out Harvard-GDP test set.*


<img width="1098" height="1000" alt="image" src="https://github.com/user-attachments/assets/bde735b9-5564-454d-9374-4decdf60f895" />

*Figure 6. Spatial activation-energy maps across the ResNet-18 stem and residual layers, together with embedding Grad-CAM maps, for the self-supervised OCT representation.*


These visualizations are intended to inspect model behaviour and the learned representation. They do not establish causality, faithfulness explanation, or clinical significance of highlighted regions.

## Repository contents and pretrained artifacts

The repository is centered on two notebooks:

- `C_01_Data_Preparation_and_OCT_SSL.ipynb` — data preparation, preprocessing, DINO pretraining, and SSL representation analysis.
- `C_02_Final_GF_Training_and_GDP_External_Evaluation.ipynb` — multimodal GF model training, operating-threshold selection, internal evaluation, and held-out GDP evaluation with MC dropout and interpretability analyses.

The following files are useful for reusing the trained models without repeating training.

| File | Purpose | Download |
|---|---|---|
| `best_model.pt` | Final trained multimodal OCT + RNFLT classifier | [Model weights](https://drive.google.com/file/d/1MI0tDAtOfZR3BjvbG2SRwdHEPr71-42t/view?usp=sharing) |
| `dino_teacher_resnet18_encoder_axis0_ssl_v2.pt` | DINO-pretrained teacher ResNet-18 encoder | [SSL encoder](https://drive.google.com/file/d/1894ouFlFkfVabSWzDVi6yB8FRui8fkkg/view?usp=sharing) |
| `rnflt_stats_gf_train_only.json` | Training-only RNFLT normalization statistics required for consistent preprocessing | [Preprocessing statistics](https://drive.google.com/file/d/105I7Agfa4lndQnk90XhI7SHyvuEoNp-B/view?usp=sharing) |
| `evaluation_protocol_locked.json` | Records the selected operating threshold and its evaluation protocol, linked to the corresponding checkpoint | [Evaluation protocol](ADD_PROTOCOL_URL) |
| `training_config.json` | Model/training configuration and provenance information | [Training configuration](https://drive.google.com/file/d/1FltTnJvsP679vXoQu2w1gVWBpShGZxPg/view?usp=sharing) |

## Running the notebooks

1. Clone or download this repository and install the required Python packages in the notebook environment.
2. Obtain access to Harvard-GF and the permitted Harvard-GDP data used by the workflow; the datasets are not included here.
3. Update dataset and output paths for your environment.
4. Run `C_01_Data_Preparation_and_OCT_SSL.ipynb` for preprocessing and SSL pretraining, or load the released SSL encoder if you do not need to repeat pretraining.
5. Run `C_02_Final_GF_Training_and_GDP_External_Evaluation.ipynb` to train the downstream classifier, or load the released classifier checkpoint for evaluation/inference.

For valid inference, keep the preprocessing consistent with training, including the 12 OCT slice indices, grayscale input preparation, RNFLT invalid-value imputation, and the **Harvard-GF training-derived** RNFLT normalization statistics. 

## Reproducibility notes

- Subject-level splits are retained for Harvard-GF.
- Harvard-GF validation/test data are not used for DINO pretraining.
- The model-selection and sensitivity-constrained threshold-selection steps use validation data only.
- The `0.404529` threshold is fixed before evaluating either test set.
- The 75 GDP test subjects are disjoint from the 500 GDP subjects used for unlabeled pretraining.
- Numerical results can still depend on checkpoint files, software versions, hardware, and deterministic-kernel settings. For exact reproduction, use the released checkpoint and matching configuration files where available.

## Paper

For the full methodology, experimental details, and complete uncertainty/interpretability results, see the accompanying manuscript:

**[Read the full paper](ADD_PAPER_URL)**

## Citation

If you use this code or the released weights, please cite the paper. 

## License and data use

This project has utilised **[Harvard-GDP](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP)** [1] and **[Harvard-GF](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF)** [2]. 

[1]. Luo, Y., Shi, M., Tian, Y., Elze, T. and Wang, M., 2023, October. Harvard glaucoma detection and progression: A multimodal multitask dataset and generalization-reinforced semi-supervised learning. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 20414-20425). IEEE.

[2]. Luo, Y., Tian, Y., Shi, M., Pasquale, L.R., Shen, L.Q., Zebardast, N., Elze, T. and Wang, M., 2024. Harvard glaucoma fairness: a retinal nerve disease dataset for fairness learning and fair identity normalization. IEEE Transactions on Medical Imaging, 43(7), pp.2623-2633. 
