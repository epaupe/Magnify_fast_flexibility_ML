import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
#to activate tensorboard logging during training, run the following command in a terminal, in the same directory as this script, and with the correct virtual environment activated:
# tensorboard --logdir runs
#then open the provided URL in a web browser

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = "/Users/edouardpaupe/Desktop/Magnify_fast_flexibility_ML"
CLIMATE_IDS = range(6)  # 0–5

FLEX_BASE_DIR = os.path.join(BASE_DIR, "data")
STATIC_FEATURE_DIR = os.path.join(BASE_DIR, "avg_scalar_params")
CLIMATE_DIR = os.path.join(BASE_DIR, "input_features/climate_scenarios")

BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
PATIENCE     = 20
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

BUILDING_IDS = [
"ep_SFH_age_0_climate_0_1241","ep_SFH_age_0_climate_0_649", "ep_SFH_age_0_climate_0_821","ep_SFH_age_0_climate_1_259", "ep_SFH_age_0_climate_1_493", "ep_SFH_age_0_climate_1_535",
"ep_SFH_age_0_climate_2_1325", "ep_SFH_age_0_climate_2_1691", "ep_SFH_age_0_climate_2_1972",
"ep_SFH_age_0_climate_3_955","ep_SFH_age_0_climate_3_1081", "ep_SFH_age_0_climate_3_1123","ep_SFH_age_0_climate_4_1072",
"ep_SFH_age_0_climate_4_1688", "ep_SFH_age_0_climate_4_1709",
"ep_SFH_age_0_climate_5_417", "ep_SFH_age_0_climate_5_758", "ep_SFH_age_0_climate_5_928",
"ep_SFH_age_1_climate_0_42", "ep_SFH_age_1_climate_0_168", "ep_SFH_age_1_climate_0_249",
"ep_SFH_age_1_climate_1_32","ep_SFH_age_1_climate_1_429", "ep_SFH_age_1_climate_1_458",
"ep_SFH_age_1_climate_2_762", "ep_SFH_age_1_climate_2_852", "ep_SFH_age_1_climate_2_1161",
"ep_SFH_age_1_climate_3_260", "ep_SFH_age_1_climate_3_451", "ep_SFH_age_1_climate_3_597"
] 

torch.manual_seed(SEED)
np.random.seed(SEED)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_climate_data(climate_id, year, month, day):
    """
    Load 48h of (sin_time, cos_time, T_amb, irrad)
    combining current day + next day if available.
    Returns array of shape (4, 192)
    """
    # --- current day ---
    fname_today = f"climate{climate_id}_{year}_{month}_{day}.csv"
    path_today = os.path.join(CLIMATE_DIR, f"climate_{climate_id}", fname_today)
    if not os.path.exists(path_today):
        print(f"⚠️ Missing climate file: {path_today}")
        return None

    df_today = pd.read_csv(path_today)
    if not {"time", "T_amb", "irrad"}.issubset(df_today.columns):
        raise ValueError(f"Invalid climate file: {fname_today}")

    # --- next day (if exists) ---
    next_day = pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(days=1)
    fname_next = f"climate{climate_id}_{next_day.year}_{next_day.month}_{next_day.day}.csv"
    path_next = os.path.join(CLIMATE_DIR, f"climate_{climate_id}", fname_next)
    if os.path.exists(path_next):
        df_next = pd.read_csv(path_next)
        df = pd.concat([df_today, df_next], ignore_index=True)
    else:
        # Pad with last day values if next day missing (end of dataset)
        pad_rows = pd.DataFrame({
            "time": pd.date_range(df_today["time"].iloc[-1], periods=96, freq="15min", inclusive="neither"),
            "T_amb": df_today["T_amb"].iloc[-1],
            "irrad": 0.0,
        })
        df = pd.concat([df_today, pad_rows], ignore_index=True)
    # --- next day loaded ---

    #sin/cos cycling encoding for training
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    hours = df["time"].dt.hour + df["time"].dt.minute / 60.0
    sin_time = np.sin(2 * np.pi * hours / 24)
    cos_time = np.cos(2 * np.pi * hours / 24)
    
    T_amb = df["T_amb"].to_numpy(dtype=np.float32)
    irrad = df["irrad"].to_numpy(dtype=np.float32)

    features = np.stack([sin_time, cos_time, T_amb, irrad], axis=0).astype(np.float32)  # (4, 192)
    if features.shape[1] != 192:
        print(f"⚠️ Unexpected climate length: {features.shape} for {fname_today}")
        return None
    #return raw dataframe for plotting (no normalization or cyclic encoding)
    df_raw = df[["time", "T_amb", "irrad"]].copy()

    return features, df_raw  # (4,192), (192, 3)


def load_flexibility_envelope(building_id, building_num, climate_id, year, month, day, base_dir=BASE_DIR):
    """
    Load a flexibility envelope (51×96) for a given building & climate & date.
    """
    building_id = f"building_{building_id}"
    flex_dir = os.path.join(base_dir, "data", building_id, "flex_env")

    fname = f"build{building_num}_clim{climate_id}_{year}_{int(month):02d}_{int(day):02d}.csv"
    fpath = os.path.join(flex_dir, fname)

    if not os.path.exists(fpath):
        print(f"⚠️ Missing envelope: {fname}")# Envelope not simulated for this building/day
        return None

    df = pd.read_csv(fpath)
    # Remove "Power Level" column if present
    if "Power Level [kW]" in df.columns:
        df = df.drop(columns=["Power Level [kW]"])
    else:
        # Some older files have no header — drop first column anyway
        df = df.iloc[:, 1:]

    arr = df.to_numpy(dtype=np.float32)

    if arr.shape[1] != 96:
        print(f"Unexpected envelope shape: {arr.shape} for {fname}")
        return None

    return arr  # (51,96)


def load_weights_and_biases(building_num, base_dir=BASE_DIR):
    """
    Loads the 61 static ARMAX scalar coefficients for a given building.
    File structure: 62×1 CSV (row 0 = building ID, rows 1–61 = coefficients).
    Returns:
        torch.tensor of shape (61,)
    """

    static_file = os.path.join(base_dir,"input_features","avg_scalar_params",f"avg_weight_and_biases_building{building_num}.csv")

    if not os.path.exists(static_file):
        raise FileNotFoundError(f"Static coefficients file not found: {static_file}")

    df = pd.read_csv(static_file, header=None)

    if df.shape[0] != 62:
        raise ValueError(
            f"Expected 62 rows (building ID + 61 coefficients), but file has: {df.shape}"
        )

    # Drop first row (contains building ID)
    coeffs = df.iloc[1:, 0].to_numpy(dtype=np.float32)

    if coeffs.shape[0] != 61:
        raise ValueError(
            f"Static feature vector must have length 61, got {coeffs.shape}"
        )

    return torch.tensor(coeffs, dtype=torch.float32)  # (61,)


def plot_weather_and_envelopes(
    pred,
    truth,
    input_features,
    means,
    stds,
    title="Flexibility Envelope Prediction",
    save_dir=None,
    file_name=None,
    show=False,
):
    """
    Plots:
      48h weather inputs (de-normalized ambient temperature + irradiance)
      Ground truth flexibility envelope
      Predicted flexibility envelope
      Signed error map (pred - true) in hours, range [-24, +24].
    """

    # --------------------------------------
    # Convert tensors → numpy
    # --------------------------------------
    pred = pred.squeeze().detach().cpu().numpy()   # (51,96)
    truth = truth.squeeze().detach().cpu().numpy() # (51,96)
    features = input_features.detach().cpu().numpy()  # (4,192)
    means = means.squeeze().cpu().numpy()
    stds = stds.squeeze().cpu().numpy()

    mae = np.mean(np.abs(pred - truth))
    mae_minutes = mae * 60.0  # convert hours → minutes

    # --------------------------------------
    # De-normalize T_amb and irrad
    # --------------------------------------
    T_amb = features[2, :] * stds[2] + means[2]
    irrad = features[3, :] * stds[3] + means[3]

    # Build continuous 48h timeline (each step = 15 min)
    time_hours = np.arange(0, 48, 0.25)

    # --------------------------------------
    # Compute signed error map (in hours)
    # --------------------------------------
    error_map = pred - truth
    error_map = np.clip(error_map, -24, 24)  # limit visualization range

    # --------------------------------------
    # Create figure (4 subplots)
    # --------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(20, 4),
                            gridspec_kw={'width_ratios': [1.5, 1, 1, 1]})
    fig.suptitle(f"{title}\nMAE = {mae:.3f} h  ({mae_minutes:.1f} min)",
                 fontsize=14, fontweight="bold")

    # 1 WEATHER INPUTS
    ax = axs[0]
    ax2 = ax.twinx()
    ax.plot(time_hours, T_amb, color="tab:red", linewidth=2, label="Ambient Temp [°C]")
    ax2.plot(time_hours, irrad, color="tab:blue", linewidth=2, alpha=0.7, label="Irradiance [W/m²]")
    ax.set_xlabel("Time [hours]")
    ax.set_xlim(0, 48)
    ax.set_xticks(np.arange(0, 49, 6))
    ax.set_ylabel("Temperature [°C]", color="tab:red")
    ax2.set_ylabel("Irradiance [W/m²]", color="tab:blue")
    ax.axvline(24, color="gray", linestyle="--", alpha=0.5)
    ax.text(12, ax.get_ylim()[1]*0.9, "Day 1", ha="center", color="gray", fontsize=9)
    ax.text(36, ax.get_ylim()[1]*0.9, "Day 2", ha="center", color="gray", fontsize=9)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)
    ax.set_title("Input Weather (48 h)")

    # 2 GROUND TRUTH ENVELOPE
    im1 = axs[1].imshow(truth, aspect="auto", origin="lower",
                        cmap="viridis")
    axs[1].set_title("Ground Truth Envelope")
    axs[1].set_xlabel("Lead time (96)")
    axs[1].set_ylabel("Power levels (51)")
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Sustained Duration [h]")

    # 3 PREDICTED ENVELOPE
    im2 = axs[2].imshow(pred, aspect="auto", origin="lower",
                        cmap="viridis", vmin=np.min(truth), vmax=np.max(truth))
    axs[2].set_title("Predicted Envelope")
    axs[2].set_xlabel("Lead time (96)")
    axs[2].set_ylabel("Power levels (51)")
    cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Sustained Duration [h]")

    # 4 SIGNED ERROR MAP (pred - true)
    im3 = axs[3].imshow(error_map, aspect="auto", origin="lower",
                        cmap="bwr", vmin=-24, vmax=24)
    axs[3].set_title("Error Map [h]\n(pred − true)")
    axs[3].set_xlabel("Lead time (96)")
    axs[3].set_ylabel("Power levels (51)")
    cbar3 = fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
    cbar3.set_label("Error [h]")

    plt.tight_layout()

    # Save or show
    if save_dir is not None and file_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path} | MAE = {mae_minutes:.1f} min")
    elif show:
        plt.show()


# =====================================================
# BUILD THE DATASET (single building, multiple climates)
# =====================================================
def get_data(base_dir=BASE_DIR,batch_size=BATCH_SIZE,seed=SEED,building_ids=BUILDING_IDS,climate_ids=CLIMATE_IDS,):
    """
    Loads:
        - climate time series  (4,192)
        - static building vector (61,)
        - flexibility envelope (1,51,96)
    for all buildings & climates. Then Normalizes, splits (85/10/5) → train/val/test DataLoaders.
    """

    X_ts_list     = []   # (4,192)
    X_static_list = []   # (61,)
    Y_list        = []   # (1,51,96)

    # ==========================================================
    # MAIN LOADING LOOP: BLD → CLIMATE → DAY
    # ==========================================================
    for building_id in building_ids:

        building_num = int(building_id.split("_")[-1])

        # Static ARMAX coefficients (61 dim)
        static_tensor = load_weights_and_biases(building_num, base_dir) # (61,)

        for clim in climate_ids:

            # Climate files for this scenario
            pattern = os.path.join(base_dir,"input_features","climate_scenarios",f"climate_{clim}",f"climate{clim}_*.csv")
            cfiles = sorted(glob.glob(pattern))
            if not cfiles:
                print(f"No climate files for climate {clim}")
                continue

            for cfile in cfiles:

                _, Y, M, D = os.path.basename(cfile).replace(".csv", "").split("_")
                year, month, day = int(Y), int(M), int(D)

                # ----- Load climate timeseries (4,192)
                result = load_climate_data(clim, year, month, day)
                if result is None:
                    continue
                X_ts, _ = result  # (4,192)

                # ----- Load flexibility envelope (1,51,96)
                Y_env = load_flexibility_envelope(
                    building_id=building_id,
                    building_num=building_num,
                    climate_id=clim,
                    year=year, month=month, day=day,
                    base_dir=base_dir
                )
                if Y_env is None:
                    continue

                # Save tensors
                X_ts_list.append(torch.tensor(X_ts, dtype=torch.float32))
                X_static_list.append(static_tensor.clone())
                Y_list.append(torch.tensor(Y_env, dtype=torch.float32).unsqueeze(0))

    # STACK ALL DATA
    X_ts_tensor     = torch.stack(X_ts_list)      # (N,4,192)
    X_static_tensor = torch.stack(X_static_list)  # (N,61) there are copies of the same building static features for each climate/day, for a total of N samples.
    Y_tensor        = torch.stack(Y_list)         # (N,1,51,96)

    print(f" Loaded dataset of length: {len(Y_tensor)} samples")
    print(f"  Climate inputs : {X_ts_tensor.shape}")
    print(f"  Static features: {X_static_tensor.shape}")
    print(f"  Envelopes      : {Y_tensor.shape}")

    # NORMALIZATION

    clim_mean = X_ts_tensor.mean(dim=(0, 2), keepdim=True)
    clim_std  = X_ts_tensor.std(dim=(0, 2), keepdim=True)
    X_ts_tensor = (X_ts_tensor - clim_mean) / (clim_std + 1e-8)

    static_mean = X_static_tensor.mean(dim=0, keepdim=True)
    static_std  = X_static_tensor.std(dim=0, keepdim=True)
    X_static_tensor = (X_static_tensor - static_mean) / (static_std + 1e-8)

    # TRAIN / VAL / TEST SPLIT
    dataset = TensorDataset(X_ts_tensor, X_static_tensor, Y_tensor)
    print(len(dataset))
    n_total = len(dataset)
    n_train = int(0.85 * n_total) # 85% for training
    n_test  = n_total - n_train

    train_set, test_set = random_split(dataset, [n_train, n_test],
                                       generator=torch.Generator().manual_seed(seed))

    n_val = int(0.15 * len(train_set)) # 15% of train set for validation
    n_train_final = len(train_set) - n_val

    train_set, val_set = random_split(train_set, [n_train_final, n_val],
                                      generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    print(f" Dataset split: {n_train_final} train | {n_val} val | {n_test} test")

    return (
        train_loader,
        val_loader,
        test_loader,
        clim_mean,
        clim_std,
        static_mean,
        static_std,
    )


# ===========================
# MODEL
# ===========================
class FlexibilityCNN(nn.Module):
    """
    (B,4,192) -> (B,1,51,96)
    1D temporal encoder -> FC bridge to small 2D latent -> 2D upsampling decoder
    """
    def __init__(self):
        super().__init__()

        # 1D TEMPORAL ENCODER (extract weather feature
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2),  # (B,32,192)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), # (B,64,192)
            nn.ReLU(),
            nn.MaxPool1d(2),                             # (B,64,96)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),# (B,128,96)
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),# (B,128,96)
            nn.ReLU(),
            nn.MaxPool1d(2),                             # (B,128,48)
        )

        # LATENT PROJECTION (1D → 2D embedding)
        self.fc = nn.Linear(128 * 48, 256 * 12)         # (B, 3072)
        
        # 2D DECODER: upsample to ~ (52,96), then trim to (51,96) in the forward pass
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=4, padding=1), # (≈13,24)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1), # (≈26,48)
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1), # (≈52,96)
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)                         # (≈52,96)
        )

    def forward(self, x):
        x = self.encoder(x)                 # (B,128,48)
        x = x.view(x.size(0), -1)           # (B, 6144), flatten
        x = self.fc(x)                      # (B, 3072)
        x = x.view(x.size(0), 256, 3, 4)    # (B,256,3,4)
        x = self.decoder(x)                 # (B,1,~52,96)
        x = F.interpolate(x, size=(51, 96), mode='bilinear', align_corners=False) #final interpolation step to (51,96)
        #    x = torch.clamp(x, 0, 24) #sustainability duration limits
        return x

class FlexibilityFusionModel(nn.Module):
    """
    CNN (time-series) + MLP (static building ARMAX coefficients)
    fused in latent space → 2D decoder → Flexibility envelope (1,51,96)
    """
    def __init__(self, static_dim=61):
        super().__init__()

        # CNN BRANCH (time-series)
        self.encoder_ts = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),      # (B,64,96)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),      # (B,128,48)
        )

        self.ts_latent_dim = 128 * 48   # 6144
        self.ts_fc = nn.Linear(self.ts_latent_dim, 1024)

        # MLP BRANCH (static coefficients)
        self.mlp = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # FUSION LAYER
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024 + 256, 3072),
            nn.ReLU(),
        )

        # 2D DECODER (same as in FlexibilityCNN)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_ts, x_static):
        # CNN branch
        h_ts = self.encoder_ts(x_ts)            # (B,128,48)
        h_ts = h_ts.view(h_ts.size(0), -1)      # (B,6144)
        h_ts = self.ts_fc(h_ts)                 # (B,1024)

        # MLP branch
        h_static = self.mlp(x_static)           # (B,256)

        # Fusion
        h = torch.cat([h_ts, h_static], dim=1)  # (B,1280)
        h = self.fusion_fc(h)                   # (B,3072)
        h = h.view(h.size(0), 256, 3, 4)        # (B,256,3,4)

        # Decoder
        out = self.decoder(h)                   # (B,1,~52,96)
        out = F.interpolate(out, size=(51, 96), mode='bilinear', align_corners=False)
        return out

# =====================================================
# TRAINING + VALIDATION LOOP
# =====================================================
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    wd=WEIGHT_DECAY,
    device=DEVICE,
    patience=PATIENCE,
    save_path="best_model.pt",
):
    """
    Train the CNN with validation monitoring, early stopping,
    checkpoint saving, and TensorBoard logging (MAE + R²).
    """

    writer = SummaryWriter(log_dir="runs/flexibility_cnn")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.L1Loss()  # MAE loss

    best_val_loss = float("inf")
    patience_counter = 0

    for ep in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        all_y_true_train, all_y_pred_train = [], []

        for X_ts, X_static, Y in train_loader:
            X_ts, X_static, Y = X_ts.to(device), X_static.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X_ts, X_static)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # collect predictions for R²
            all_y_true_train.append(Y.detach().cpu().numpy().flatten())
            all_y_pred_train.append(preds.detach().cpu().numpy().flatten())

        train_loss /= len(train_loader)
        y_true_train = np.concatenate(all_y_true_train)
        y_pred_train = np.concatenate(all_y_pred_train)
        r2_train = r2_score(y_true_train, y_pred_train)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        all_y_true_val, all_y_pred_val = [], []

        with torch.no_grad():
            for X_ts, X_static, Y in val_loader:
                X_ts, X_static, Y = X_ts.to(device), X_static.to(device), Y.to(device)
                preds = model(X_ts, X_static)
                val_loss += criterion(preds, Y).item()

                all_y_true_val.append(Y.detach().cpu().numpy().flatten())
                all_y_pred_val.append(preds.detach().cpu().numpy().flatten())

        val_loss /= len(val_loader)
        y_true_val = np.concatenate(all_y_true_val)
        y_pred_val = np.concatenate(all_y_pred_val)
        r2_val = r2_score(y_true_val, y_pred_val)

        # ---- LOG TO TENSORBOARD ----
        writer.add_scalar("MAE/Train", train_loss, ep)
        writer.add_scalar("MAE/Validation", val_loss, ep)
        writer.add_scalar("R2/Train", r2_train, ep)
        writer.add_scalar("R2/Validation", r2_val, ep)

        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            status = "Saved best model"
        else:
            patience_counter += 1
            status = f"No improvement ({patience_counter}/{patience})"

        writer.add_text("Status", f"Epoch {ep}: {status}")
        print(f"Epoch {ep:03d} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | R² Train: {r2_train:.4f} | R² Val: {r2_val:.4f} | {status}")

        if patience_counter >= patience:
            print(f"Early stopping after {ep} epochs (no improvement for {patience} epochs).")
            break

    writer.close()
    print(f"Best validation MAE: {best_val_loss:.4f} (model saved at '{save_path}')")
    model.load_state_dict(torch.load(save_path))
    return model

def test_model(model, test_loader, device=DEVICE, results_dir=None):
    """
    Evaluate trained model on test set:
      - Mean Absolute Error (MAE)
      - R² coefficient
      - Average computation time per flexibility envelope (s/envelope)
    Also logs metrics to TensorBoard and optionally saves them as .txt file.
    """
    model.eval()
    criterion = nn.L1Loss()

    test_loss = 0.0
    all_y_true, all_y_pred = [], []
    total_time = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_ts, X_static, Y in test_loader:
            X_ts, X_static, Y = X_ts.to(device), X_static.to(device), Y.to(device)

            start_time = time.time()
            preds = model(X_ts, X_static)
            preds = torch.clamp(preds, 0, 24)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            n_samples += X_ts.size(0)

            test_loss += criterion(preds, Y).item()

            all_y_true.append(Y.detach().cpu().numpy().flatten())
            all_y_pred.append(preds.detach().cpu().numpy().flatten())

    # Average MAE over all batches
    test_loss /= len(test_loader)

    # Concatenate all predictions/targets for R²
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    r2 = r2_score(y_true, y_pred)

    # Average computation time per envelope (s)
    avg_time_per_sample = total_time / n_samples if n_samples > 0 else 0.0

    print(f"Test MAE: {test_loss:.5f} h")
    print(f"Test MAE: {test_loss * 60.0:.2f} minutes")
    print(f"Test R²:  {r2:.5f}")
    print(f"Average computation time: {avg_time_per_sample:.5f} s/envelope")

    # Log to TensorBoard
    writer = SummaryWriter(log_dir="runs/flexibility_cnn_test")
    writer.add_scalar("Test/MAE", test_loss)
    writer.add_scalar("Test/R2", r2)
    writer.add_scalar("Test/Computation_Time_s_per_envelope", avg_time_per_sample)
    writer.close()

    # Save to text file if results_dir is provided
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, "test_set_results_metrics.txt")
        with open(file_path, "w") as f:
            f.write("Test Set Evaluation Metrics\n")
            f.write("===========================\n")
            f.write(f"Test MAE [h]: {test_loss:.5f}\n")
            f.write(f"Test MAE [min]: {test_loss * 60.0:.2f}\n")
            f.write(f"Test R²: {r2:.5f}\n")
            f.write(f"Avg Computation Time [s/envelope]: {avg_time_per_sample:.5f}\n")
        print(f"Saved test metrics to {file_path}")

    return test_loss, r2, avg_time_per_sample

def main():
    train_loader, val_loader, test_loader, clim_mean, clim_std, static_mean, static_std = get_data()
    model = FlexibilityFusionModel()
    # model = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     epochs=EPOCHS,
    #     lr=LR,
    #     wd=WEIGHT_DECAY,
    #     device=DEVICE,
    #     patience=PATIENCE,
    #     save_path="best_flex_fusion.pt",
    # )
    # or: load a pre-trained model
    model.load_state_dict(torch.load("best_flex_fusion.pt", map_location=DEVICE))

    # Save test predictions plots
    results_dir = os.path.join(BASE_DIR, "ML_PIPELINE", "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving test predictions to: {results_dir}")

    test_model(model, test_loader, device=DEVICE, results_dir=results_dir)

    model.eval()
    with torch.no_grad():
        sample_idx = 0  # global counter across all batches
        for batch_idx, (X_ts_batch, X_static_batch, Y_batch) in enumerate(test_loader): #iterates over every batch in the test set
            preds = model(X_ts_batch.to(DEVICE), X_static_batch.to(DEVICE))  # (B,1,51,96)
            preds = torch.clamp(preds, 0, 24)  #sustainability duration limits

            for j in range(X_ts_batch.size(0)):  # loop over batch samples
                X_sample = X_ts_batch[j]
                Y_true = Y_batch[j]
                Y_pred = preds[j].cpu()

                plot_weather_and_envelopes(
                    pred=Y_pred,
                    truth=Y_true,
                    input_features=X_sample,
                    means=clim_mean,
                    stds=clim_std,
                    title=f"Flexibility Envelope Prediction of Test Sample {sample_idx} — from Test Set",
                    save_dir=results_dir,
                    file_name=f"prediction{sample_idx}_test_set.png",
                    show=False
                )

                sample_idx += 1

    print(f"✅ Saved {sample_idx} prediction plots to '{results_dir}'")
    return None

if __name__ == "__main__":
    main()
