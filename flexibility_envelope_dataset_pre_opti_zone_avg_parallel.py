"""
CREATE FLEXIBILITY ENVELOPE DATASET
-----------------------------------
Compute and store flexibility envelopes for all buildings and climates over a specified
time window. For each (building_id, climate_id), the script:
  1. Runs the MPC optimization to obtain episode-wise UB/LB power bounds
  2. Extracts daily UB/LB arrays using extract_daily_building_bounds()
  3. Computes and saves flexibility envelopes + heatmaps

Author: Edouard Paupe
Date: 2025-10-20
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

# Enforce single-threaded BLAS operations to fully utilize logical cores efficiently
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.env import Env
from src.flex import envelope_for_zone_day
from src.utils import compute_avg_armax, make_scalar_armax_config
from src.agents_scalar import MPCScalar, RB

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = r"C:\Users\palo\magnify-main_DATABASE_SCALAR"

START_TIME = datetime.datetime(2020, 2, 2, 0, 0, 0)
END_TIME   = datetime.datetime(2020, 2, 3, 0, 0, 0)
HISTORY_HOURS = 8
HORIZON_HOURS = 24
STEPS_PER_HOUR = 4
HP_POWER = 1  # [kW] we assume normalized power
CLIMATE_IDS = range(6)  # 0–5
N_CORES = 16  # Number of logical CPU cores to use for parallelization

# Building archetypes 
BUILDING_IDS = [
    "ep_SFH_age_0_climate_0_1241",
    "ep_SFH_age_0_climate_0_649", "ep_SFH_age_0_climate_0_821", "ep_SFH_age_0_climate_0_1241",
    "ep_SFH_age_0_climate_1_259", "ep_SFH_age_0_climate_1_493", "ep_SFH_age_0_climate_1_535",
    "ep_SFH_age_0_climate_2_1325", "ep_SFH_age_0_climate_2_1691", "ep_SFH_age_0_climate_2_1972",
    "ep_SFH_age_0_climate_3_955", "ep_SFH_age_0_climate_3_1081", "ep_SFH_age_0_climate_3_1123",
    "ep_SFH_age_0_climate_4_1072", "ep_SFH_age_0_climate_4_1688", "ep_SFH_age_0_climate_4_1709",
    "ep_SFH_age_0_climate_5_417", "ep_SFH_age_0_climate_5_758", "ep_SFH_age_0_climate_5_928",
    "ep_SFH_age_1_climate_0_42", "ep_SFH_age_1_climate_0_168", "ep_SFH_age_1_climate_0_249",
    "ep_SFH_age_1_climate_1_32", "ep_SFH_age_1_climate_1_429", "ep_SFH_age_1_climate_1_458",
    "ep_SFH_age_1_climate_2_762", "ep_SFH_age_1_climate_2_852", "ep_SFH_age_1_climate_2_1161",
    "ep_SFH_age_1_climate_3_260", "ep_SFH_age_1_climate_3_451", "ep_SFH_age_1_climate_3_597"
]

# =====================================================
# STEP 1 — OPTIMIZATION: Run MPC and retrieve UB/LB arrays
# =====================================================

def compute_episode_power_bounds(env, hp_power=HP_POWER):
    """
    Single MPC on the average-zone dynamics (scalar ARMAX).
    Returns UB/LB arrays of shape (n_episodes, horizon_length)
    """
    # Build averaged (scalar) ARMAX config from the env's multi-zone model
    avg_params   = compute_avg_armax(env.armax_config)
    scalar_config = make_scalar_armax_config(
        avg_params,
        history_length=env.history_length,
        horizon_length=env.horizon_length
    )

    # Temps bounds matrices now  (horizon+1, 1) since 1 zone
    T_min = np.full((env.horizon_length + 1, 1), 20.0)
    T_max = np.full((env.horizon_length + 1, 1), 22.0)

    upper_bound = MPCScalar(
        avg_config=scalar_config, target_temperature=21.0,
        T_min=T_min, T_max=T_max,
        history_length=env.history_length, horizon_length=env.horizon_length,
        objective="upper_bound"
    )
    lower_bound = MPCScalar(
        avg_config=scalar_config, target_temperature=21.0,
        T_min=T_min, T_max=T_max,
        history_length=env.history_length, horizon_length=env.horizon_length,
        objective="lower_bound"
    )

    # Rule-based baseline still ok (it returns per-zone actions; env.step averages through physics anyway)
    controller = RB(n_zones=env.n_zones, T_min=20.0, T_max=22.0)

    obs, _ = env.reset()
    while not env.terminated:
        upper_bound.predict(obs)
        lower_bound.predict(obs)
        upper_bound.save_episode()
        lower_bound.save_episode()

        actions = controller.predict(obs)  # baseline action for Env propagation
        obs, _, _, _, _ = env.step(actions)

    # Collect episodes → each is single zone ⇒ we flatten directly
    n_episodes = len(upper_bound.results["control_action"])
    horizon_length = env.horizon_length

    #array initialization
    upper_power_bound = np.zeros((n_episodes, horizon_length))
    lower_power_bound = np.zeros((n_episodes, horizon_length))
    T0 = np.zeros((n_episodes, 1))  # average initial temp (for completeness)

    for t in range(n_episodes):
        u_upp = np.array([list(x.values()) for x in upper_bound.results["control_action"][t].values()])  # (H+1, 1)
        u_low = np.array([list(x.values()) for x in lower_bound.results["control_action"][t].values()])   # (H+1, 1)
        _T0   = np.array(list(lower_bound.results["temperature"][t][0].values()))  # (1,)

        u_upp = u_upp[env.history_length:, :]  # (horizon_length, 1)
        u_low = u_low[env.history_length:, :]

        upper_power_bound[t, :] = u_upp[:, 0] * hp_power
        lower_power_bound[t, :] = u_low[:, 0] * hp_power
        T0[t, 0] = _T0

    print("Upper power bound shape:", upper_power_bound.shape)
    print("Lower power bound shape:", lower_power_bound.shape)
    print("Initial avg temperature shape:", T0.shape)

    return upper_power_bound, lower_power_bound, T0


# =====================================================
# STEP 2 — Extract daily bounds: separates the optimization results into daily (96,96) chunks
# =====================================================

def extract_daily_building_bounds(upper_power_bound, lower_power_bound, env, save_dir=None):
    """
    Extract daily UB/LB arrays (96 x horizon_length) for each day, aggregated across all zones.
    """
    n_episodes, n_horizon = upper_power_bound.shape
    block_size = 96  # 1 day = 96 episodes of 15 min
    n_blocks = n_episodes // block_size

    daily_bounds = {}
    building_num = int(env.building_id.split('_')[-1])
    climate_id = env.climate_id

    for b in range(n_blocks):
        start_idx = b * block_size
        end_idx = start_idx + block_size

        ub_chunk = upper_power_bound[start_idx:end_idx, :]
        lb_chunk = lower_power_bound[start_idx:end_idx, :]

        current_date = (env.start_time + datetime.timedelta(days=b)).date()
        day = current_date.day
        month = current_date.month
        year = current_date.year

        ub_name = f"build{building_num}_clim{climate_id}_{year}_{month:02d}_{day:02d}_UB"
        lb_name = f"build{building_num}_clim{climate_id}_{year}_{month:02d}_{day:02d}_LB"

        daily_bounds[ub_name] = ub_chunk
        daily_bounds[lb_name] = lb_chunk

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ub_file_path = os.path.join(save_dir, f"{ub_name}.npy")
            lb_file_path = os.path.join(save_dir, f"{lb_name}.npy")
            
            # Check if files already exist
            if not os.path.exists(ub_file_path):
                np.save(ub_file_path, ub_chunk)
            else:
                print(f"File already exists, skipping: {ub_file_path}")

            if not os.path.exists(lb_file_path):
                np.save(lb_file_path, lb_chunk)
            else:
                print(f"File already exists, skipping: {lb_file_path}")

    print(f"Extracted {n_blocks} daily UB/LB pairs for building {building_num}, climate {climate_id}")
    return daily_bounds


# =====================================================
# STEP 3 — Compute and save flexibility envelopes
# =====================================================

def save_flexibility_envelopes(bounds_dict, flex_env_dir, flex_img_dir):
    os.makedirs(flex_env_dir, exist_ok=True)
    os.makedirs(flex_img_dir, exist_ok=True)

    for ub_name, ub_chunk in bounds_dict.items():
        if not ub_name.endswith("_UB"):
            continue
        base_name = ub_name.replace("_UB", "")
        lb_chunk = bounds_dict.get(f"{base_name}_LB")
        if lb_chunk is None:
            continue

        csv_path = os.path.join(flex_env_dir, f"{base_name}.csv")
        if os.path.exists(csv_path):
            print(f"Flexibility envelope already exists, skipping: {csv_path}")
            continue  # Skip envelope if already computed

        P_grid, durations = envelope_for_zone_day(ub_chunk, lb_chunk, dt_h=1/4, P_min=0, P_max=1.0, dP=0.02)

        # Save CSV
        df = pd.DataFrame(
            durations,
            index=np.round(P_grid, 2),
            columns=[f"LeadTime_{i}" for i in range(durations.shape[1])]
        )
        df.index.name = "Power Level [kW]"
        df.to_csv(csv_path)

        # Save heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            durations, aspect="auto", origin="lower",
            extent=[0, durations.shape[1]*0.25, P_grid[0], P_grid[-1]],
            cmap="viridis", vmin=0, vmax=24
        )
        plt.colorbar(im, ax=ax, label="Max sustained duration [h]")
        ax.set_xlabel("Lead time [h]")
        ax.set_ylabel("Power [kW]")
        ax.set_title(f"Flexibility Envelope — {base_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(flex_img_dir, f"{base_name}.png"), dpi=300)
        plt.close(fig)


# =====================================================
# PARALLELIZED FUNCTION — handles each (building, climate) pair
# =====================================================

def process_building_climate(args):
    building_id, climate_id, start_time_global = args
    try:
        # 0 create building directories
        building_folder = os.path.join(BASE_DIR, f"data/building_{building_id}")
        power_bounds_dir = os.path.join(building_folder, "power_bounds")
        flex_env_dir = os.path.join(building_folder, "flex_env")
        flex_img_dir = os.path.join(building_folder, "flex_env_images")
        os.makedirs(power_bounds_dir, exist_ok=True)
        os.makedirs(flex_env_dir, exist_ok=True)
        os.makedirs(flex_img_dir, exist_ok=True)    

        # =====================================================
        # DAILY LOOP: from START_TIME → END_TIME
        # =====================================================
        total_days = (END_TIME - START_TIME).days

        for day_idx in range(total_days):
            day_start = START_TIME + datetime.timedelta(days=day_idx)
            day_end = day_start + datetime.timedelta(days=1)
            date_str = day_start.strftime("%Y_%m_%d")

            print(f"\n---- {building_id} | Climate {climate_id} | Day {day_idx + 1}/{total_days}: {day_start.date()} ----")

            # File paths for this day
            ub_file = os.path.join(power_bounds_dir, f"build{building_id.split('_')[-1]}_clim{climate_id}_{date_str}_UB.npy")
            lb_file = os.path.join(power_bounds_dir, f"build{building_id.split('_')[-1]}_clim{climate_id}_{date_str}_LB.npy")
            csv_file = os.path.join(flex_env_dir, f"build{building_id.split('_')[-1]}_clim{climate_id}_{date_str}.csv")

            # Skip if both power bounds and flexibility envelope already exist
            if os.path.exists(csv_file):
                print(f"Already processed {building_id} | Climate {climate_id} | {date_str}, skipping.")
                continue

            start_time_local = time.time()

            # Initialize one-day environment
            env = Env(
                building_id=building_id,
                climate_id=climate_id,
                start_time=day_start,
                end_time=day_end,
                history_hours=HISTORY_HOURS,
                horizon_hours=HORIZON_HOURS,
                steps_per_hour=STEPS_PER_HOUR,
            )

            # 1 - Compute or load daily UB/LB arrays
            if os.path.exists(ub_file) and os.path.exists(lb_file):
                print("Power bounds already exist, loading from disk...")
                ub = np.load(ub_file)
                lb = np.load(lb_file)
            else:
                print("Computing new power bounds...")
                ub, lb, T0 = compute_episode_power_bounds(env)
                bounds_dict = extract_daily_building_bounds(ub, lb, env, save_dir=power_bounds_dir)
            # 2 - Compute and save daily flexibility envelopes
            if not os.path.exists(csv_file):
                bounds_dict = extract_daily_building_bounds(ub, lb, env)
                save_flexibility_envelopes(bounds_dict, flex_env_dir, flex_img_dir)

            # Timing info for this day
            print(f"Time for {building_id} | Climate {climate_id} | {day_start.date()}: {(time.time() - start_time_local):.2f} s")
            print(f"Total elapsed so far: {(time.time() - start_time_global)/60:.2f} min")

    except Exception as e:
        print(f"⚠️ Skipped {building_id} (climate {climate_id}) due to error:\n{e}")


# =====================================================
# MAIN PIPELINE (DAILY EXECUTION + PARALLELIZATION + TIMING)
# =====================================================

if __name__ == "__main__":
    print("os.cpu_count() =", os.cpu_count())
    start_time_global = time.time()  # Total timer

    # Create all (building, climate) combinations
    tasks = [(building_id, climate_id, start_time_global) for building_id in BUILDING_IDS for climate_id in CLIMATE_IDS]
    print(f"\nLaunching parallel computation on {len(tasks)} tasks using {N_CORES} logical cores...")

    # Run parallel execution using multiprocessing
    with mp.Pool(processes=N_CORES) as pool:
        pool.map(process_building_climate, tasks)

    print("\n✅ All flexibility envelopes processed successfully!")
    print(f"Total elapsed time: {(time.time() - start_time_global)/60:.2f} minutes ({time.time() - start_time_global:.1f} seconds)")
