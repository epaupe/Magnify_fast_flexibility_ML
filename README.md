# **Machine Learning-Assisted Large Scale Quantification of Building Energy Flexibility**
**MAGNIFY: Data-Driven Demand-Side Flexibility Quantification**

---

## **Project Overview**
This repository provides a complete computational framework to **quantify and predict building demand-side flexibility**. It unifies:

- An optimization-based MPC pipeline that simulates building thermal dynamics and computes ground-truth flexibility envelopes.

- A deep-learning ML pipeline (PyTorch), with a complete CNN+MLP multi-modal architecture that predicts complete flexibility envelopes in real time from weather forecasts and building parameters.

The goal is to replace slow, repeated optimization with a single-shot ML prediction, enabling scalable real-time flexibility quantification for grid-aware control and flexibility markets.

## **Output Example**
![Expected result of ML model output on test set](ML_PIPELINE/prediction_example.png)

---

# **1. Optimization-Based Flexibility Envelope Dataset Generation**  
*(flexibility_envelope_dataset_parallel.py)*

This script computes **ground-truth flexibility envelopes** for all buildings, climates, and days using a highly parallelized MPC workflow.
The script produces all training labels for the ML supervised learning pipeline:
- 30 Buildings x 6 climates x 365 days = 65700 training flexibility envelopes.

---
## **Workflow Overview**

For each `(building_id, climate_id)` pair, the script:

1. **Simulates daily building operation using scalar ARMAX dynamics** derived from multi-zone models.
2. **Runs two MPC controllers**:  
   - *Upper-bound controller* (maximum feasible power)  
   - *Lower-bound controller* (minimum feasible power)
3. **Extracts daily UB/LB matrices** (96 episodes × 96 horizon steps) using  
   `extract_daily_building_bounds()`.
4. **Computes flexibility envelopes** using  
   `envelope_for_zone_day()`, producing a matrix of:  
   - 51 discrete **power levels**  
   - 96 **lead times**  
   Each entry gives the **maximum sustained duration** (in hours).
5. **Saves results** as:  
   - `.csv` envelope (51 × 96 matrix)  
   - `.png` heatmap  
   stored under:  
   ```text
   data/building_{building_name}/flex_env/
   data/building_{building_name}/flex_env_images/

--- 

# **2. Machine Learning Prediction Pipeline**

This repository contains **two complementary machine-learning pipelines** for predicting full flexibility envelopes using a CNN–MLP fusion architecture.  
Both pipelines share the same model architecture and data format, but differ in their **generalization objective and dataset split strategy**.

---

## **Implemented Scripts**

### **1. `multi_building_ML_prediction_new_dataset.py`**  
**Generalization to unseen weather scenarios**

This script trains and evaluates the model on the full set of **82 ARMAX building archetypes**, while enforcing a dataset split that ensures **unseen weather conditions** in the test set.

**Key characteristics:**
- All 82 ARMAX building parameter sets are used during training.
- Train, validation, and test splits are performed **across weather scenarios only**.
- Building parameters appearing in the test set have already been seen during training.
- In this setting, ARMAX coefficients effectively act as a **building identifier**.

**Underlying assumption:**  
The set of 80+ ARMAX building archetypes is assumed to be representative of a large population of real-world residential buildings.  
Therefore, any building provided to the ML predictor is assumed to correspond to a building archetype already seen during training, and the model’s task is to infer flexibility envelopes under **previously unseen climatic conditions**.

---

### **2. `multi_building_ML_prediction_gen_unseen_new_dataset.py`**  
**Generalization to unseen buildings and unseen weather**

This script evaluates the model’s ability to generalize simultaneously to:
- unseen weather scenarios, and
- unseen building dynamics.

To achieve this, buildings themselves are explicitly held out during testing.

**Building-level split:**
- **61 buildings** for training  
- **10 buildings** for validation  
- **11 buildings** for testing  

In this configuration:
- ARMAX parameters in the test set have **never been observed** during training.
- The model must learn a mapping from building dynamics and weather inputs to flexibility envelopes without relying on prior exposure to the same building.

This scenario represents a stricter and more realistic generalization setting, where the ML model is applied to **new buildings with unknown dynamic characteristics**.

---

## **Input Features (common to both pipelines)**

Each sample consists of:

### **1. Weather time series (4 × 192)**  
- `sin_time`  
- `cos_time`  
- `T_amb`  
- `irradiance`  

→ 48 hours sampled at 15-minute resolution.

### **2. Static building parameters (85)**  
ARMAX scalar coefficients extracted from the building models.

These coefficients are **time-invariant descriptors** of the building dynamics and encode information related to:
- thermal mass and inertia  
- envelope insulation  
- geometry and effective surface areas  
- internal zoning and heat transfer characteristics  
- additional control-relevant dynamics captured by the extended ARMAX formulation  

### **3. Ground-truth flexibility envelope (1 × 51 × 96)**  
Generated by the MPC-based optimization pipeline.

---

## **Model Architecture: `FlexibilityFusionModel`**

The architecture is identical in both scripts and consists of:

- **CNN branch**: encodes the 48-hour weather sequence  
- **MLP branch**: encodes static ARMAX building parameters  
- **Fusion layer**: combines dynamic and static latent representations  
- **2D convolutional decoder**: reconstructs the full flexibility envelope  

The network predicts **all 4 896 envelope values (51 × 96) in a single forward pass**.

The MLP branch input dimension is set to **85** to match the extended ARMAX parameter vector.

---

## **Data Pipeline**

For both scripts, the workflow is:

1. Load weather time series, building parameters, and envelope labels  
2. Perform scenario-specific train, validation, and test splitting  
3. Compute normalization statistics from the **training set only**  
4. Build normalized PyTorch `DataLoader`s  
5. Enforce **strict separation** between training and evaluation data to avoid data leakage

---

## **Training Workflow**

- **Loss function**: MAE (L1)  
- **Optimizer**: AdamW  
- **Early stopping** based on validation loss  
- **TensorBoard logging**:
  - training and validation MAE  
  - training and validation R²  
  - epoch summaries and model checkpoints  
- Automatic saving of the **best-performing model**

---

## **Testing and Visualization**

The evaluation phase reports:

- Test **MAE** (in hours and minutes)  
- Test **R² score**  
- **Average inference time** per flexibility envelope  

Additionally, up to **1 000 visualization plots** are generated, each showing:
- input weather conditions  
- ground-truth flexibility envelope  
- predicted flexibility envelope  
- signed prediction error map  

---
# **3. ARMAX Parameters Extraction**

The extraction of static building parameters is handled by the script  
`scalar_weight_and_biases.py`.

This script processes the set of ARMAX model **weights and biases** stored in the
`armax_models/` directory, across all climate zones associated with a given building archetype.

---

## **Extraction Procedure**

For each building archetype, the script:

1. Loads all ARMAX model weights and biases available across climate zones  
2. Aggregates these parameters at the **building level** by computing averages across zones  
3. Constructs a **fixed-size static feature vector of shape (85,)**  
4. Saves the resulting vector to:
'input_features/avg_scalar_params/'

These averaged vectors are subsequently used as the **static building parameters** in the machine-learning prediction pipelines.

___
## **WARNING on Dimensionality and Lag Dependency**

The dimensionality of the extracted vector, currently **85**, is **entirely determined by the lag structure** of the ARMAX models.

Specifically, the size depends on the list of temporal lags used when extracting the weights and biases.

At present, the lag configuration is **hard-coded** in the function  
`load_armax_model` located in:

`src/utils.py`

with the following definition:

lag_hours_list = [6/12, 9/12, 1, 2, 4, 8]

Each lag corresponds to a specific group of ARMAX coefficients, and the concatenation of all associated weights and biases across these lags defines the final feature vector size.

## **Important Consistency Requirement**

If the ARMAX lag configuration is modified, the following components must be updated consistently:

the lag definition in src/utils.py

the ARMAX parameter extraction logic in scalar_weight_and_biases.py

the expected input dimensionality of the MLP branch in the ML models

Failure to update all components accordingly will result in a dimensionality mismatch between extracted features and the trained machine-learning model.

---

# **4. Repository Structure & Folder Descriptions**

This repository is organized into modular components that jointly support the MPC-based dataset generation and the ML-based flexibility envelope prediction pipeline.

---

## **📁 armax_models/**
This folder contains:

- The **ARMAX building archetypes**, each describing a building’s dynamic response through:
  - Structural coefficients  
  - Thermal response coefficients  
  - Intercepts  
- Raw **MeteoSwiss weather scenario data**, before any preprocessing.
- These files constitute the **physical model inputs** used both for:
  - The MPC envelope generation pipeline  
  - The extraction of static input features for the ML pipeline  

---

## **📁 data/**
Contains the **computed optimization results**—the full set of flexibility envelope training labels generated by:
flexibility_envelope_dataset_parallel.py
For each building and climate, it stores:

- Daily **upper/lower power bounds** 
- Daily **flexibility envelope matrices** (51 × 96)  
- Optional heatmap visualizations  

---

## **📁 input_features/**
Contains the **processed input features** for the ML model:

### Time-series features:
- Processed climate scenarios for each climate zone  
- 48h weather sequences (sin_time, cos_time, ambient temperature, irradiance)

### Static features:
- Each building’s **ARMAX scalar coefficients** (weights & biases), compressed into a 61-dimensional static descriptor used by the MLP branch of the model.

All files in this folder are produced by the notebooks in `notebooks/`.

---

## **📁 notebooks/**
Contains analysis and feature extraction notebooks, including:

- **features_extraction.ipynb**  
  Extracts and formats time-series input features from raw climate scenario files.

This notebook populates the **input_features/** directory.

---

## **📁 ML_PIPELINE/**
This folder aggregates all ML outputs.
Contains:

- Saved **trained model weights** (e.g., `best_flex_fusion.pt`)  
- Complete **test-set evaluation results**  
- Prediction plots and metric summaries from the ML pipeline 

---

## **📁 runs/**
Contains the **TensorBoard logs** generated during training and testing:

- MAE curves (train/val/test)
- R² curves (train/val/test)
- Training status and metadata

To visualize:  
tensorboard --logdir runs

---
## **📁 src/**
This folder contains all core Python modules required to run the MPC-based flexibility envelope generation pipeline.  
These files implement the building environment, MPC controllers, ARMAX processing utilities, and envelope construction logic.

More info is available in the dedicated README.md file. 

---

# **5. Repository Data Availability**

Due to **GitLab repository memory limitations**, several large folders are **not tracked** in this repository and have been explicitly ignored.

The following directories are **not included** in the GitLab repository:
- `armax_models`
- `data`
- `ML_pipeline`

These folders are required to fully reproduce the experiments, retrain the models, or rerun the prediction pipelines.

---

## **Manual Download Required**

All omitted folders are available for manual download via the following Polybox link:

https://polybox.ethz.ch/index.php/s/XDBd2mt8YBZzKsS

After downloading, the folders must be placed at the **root level of the repository**, preserving their original directory structure.

---
# Authors and acknowledgment

Edouard Paupe
Master’s student in Energy Science & Technology (ETH Zürich)
Developer of the machine-learning framework, data pipelines, and multi-building flexibility-quantification models.

I would like to express my sincere gratitude to Dr. Mina Montazeri from EMPA’s Urban Energy Systems Laboratory for her continuous guidance, support, and scientific insights throughout this project.
I am also very thankful to Julie Rousseau from EMPA’s UESL for her valuable feedback, discussions, and assistance, which greatly contributed to the development and refinement of this work.

