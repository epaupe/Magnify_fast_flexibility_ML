# **Source Code Overview (`src/` Directory)**  
This folder contains all core modules required for the **optimization-based flexibility envelope generation pipeline**.  
These scripts define the thermal environment, MPC controllers, ARMAX processing utilities, and envelope computation functions used by:
*flexibility_envelope_dataset_parallel.py*

The modules work together to simulate building thermal dynamics, compute upper/lower feasible power bounds, and generate daily flexibility envelopes.

---

## **📄 env.py — Building Thermal Environment**
Defines the `Env` class, which provides:

- Thermal state evolution using **ARMAX-based** building dynamics  
- Handling of internal/external temperatures, irradiance, and disturbances  
- Looping through daily simulation time (96 × 15 min steps)
- Comfort constraints (e.g., 20–22 °C)
- Multi-zone → scalar-zone averaging when needed
- Interfaces for interaction with MPC/RB controllers (`step()`, `reset()`)

This environment is the underlying simulation model used by all MPC agents.

---

## **📄 agents_scalar.py — MPC and Rule-Based Controllers**
Contains controllers used to generate **upper-bound** and **lower-bound** feasible power trajectories:

### **MPCScalar**
A scalar MPC agent that solves two optimization problems:
- `objective="upper_bound"` → maximizes heating/cooling power  
- `objective="lower_bound"` → minimizes heating/cooling power  

It uses:
- ARMAX parameters (scalar form)
- Temperature constraints from `Env`
- Thermal predictive model over a horizon (typically 24 h × 4 steps/hour)

### **RB (Rule-Based Controller)**
A baseline controller used to propagate the environment forward while MPC agents compute UB/LB trajectories.

---

## **📄 mpc_scalar.py — Scalar MPC Optimization Logic**
Implements the scalar MPC optimization problem, including:

- Cost function definition for UB/LB objectives  
- Temperature soft constraints  
- Power and comfort limits  
- Horizon unrolling and solver execution  

This module contains the numerical optimization routines used by `MPCScalar`.

---

## **📄 flex.py — Flexibility Envelope Computation**
Provides functions to convert UB/LB power bounds into **flexibility envelopes**, including:

### **`envelope_for_zone_day()`**
Given:
- UB matrix (96 episodes × horizon)
- LB matrix (96 episodes × horizon)

It computes:
- Discrete **power grid levels** (typically 51 levels)
- For each level and lead time, the **maximum sustained duration**  
  a building can maintain that power without violating constraints.

Outputs:
- Envelope matrix → (power_levels × lead_times)

This is the core algorithm producing the final envelope dataset.

---

## **📄 plot_energy_bounds.py — Visualization Utilities**
Helpers for plotting:
- UB/LB power trajectories
- Daily MPC bound profiles  
Useful for debugging and validating MPC behavior.

Plots can be saved or displayed interactively.

---

## **📄 utils.py — Miscellaneous Utilities**
Collection of helper functions used throughout the pipeline, including:

- **ARMAX averaging** (`compute_avg_armax`)  
  Converts multi-zone ARMAX models into scalar versions.
- **Model packaging** (`make_scalar_armax_config`)  
  Formats ARMAX matrices for MPC input.
- File handling, temperature formatting, data reshaping, etc.

This module centralizes shared low-level utilities.


