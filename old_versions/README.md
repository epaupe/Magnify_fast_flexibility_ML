# **Archived Code & Legacy Versions (`old_versions/` Directory)**

This directory contains older or preliminary implementations of the dataset-generation and machine-learning pipelines.  
These files are preserved for **debugging, reference, and reproducibility**, but are not used in the current main workflow.

---

## **📄 flexibility_envelope_dataset_pre_opti_zone_avg.py**

This script is an earlier version of the dataset-generation pipeline.  
It builds the flexibility envelope dataset in the **same functional way** as:
*flexibility_envelope_dataset_parallel.py*

…but **without parallelization**.

### **Purpose**
- Serves as a **sequential baseline** for understanding envelope generation logic  
- Useful for **debugging**, testing, or validating MPC behavior step-by-step  
- Allows users to inspect the algorithm in a simpler, single-process execution flow  

### **Limitations Compared to the Current Version**
- Much **slower**, since it runs MPC simulations for each building and day **one after another**
- Cannot scale efficiently to the full training dataset  
- Replaced by the parallel version to fully utilize modern multi-core CPUs

---

## **📄 single_building_training_dataset.py**

This script implements the **first prototype** of the ML prediction pipeline.  
It trains a CNN model **only on a single building**, while varying climate scenarios.

### **Purpose**
- First experimental step of the project  
- Validates whether a CNN can learn to predict flexibility envelopes *from weather inputs alone*  
- Establishes a baseline before scaling to multi-building generalization

### **Model Characteristics**
- Inputs:  
  - 48-hour weather time series (sin/cos time, T_amb, irradiance)
- Output:  
  - A predicted flexibility envelope for **one specific building**
- No static ARMAX parameters (weights/biases) used yet

### **Role in the Project**
- Demonstrates feasibility of envelope prediction using ML  
- Provides insights into CNN architecture tuning  
- Serves as a **stepping stone** toward the final multi-building, multi-climate model using both dynamic and static inputs
