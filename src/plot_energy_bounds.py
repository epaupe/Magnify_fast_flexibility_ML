import os
import matplotlib.pyplot as plt
from flex import (plot_energy_bounds_from_file)

BASE_DIR = "/Users/edouardpaupe/Desktop/Magnify_fast_flexibility_ML"
try:
    fig, ax = plot_energy_bounds_from_file(
        building_id="ep_SFH_age_0_climate_0_1241",
        building_num=1241,
        climate_id=0,
        year=2020,
        month=1,
        day=3,
        ep_idx=20,
        base_dir=BASE_DIR,
        steps_per_hour=4
    )
    plt.show()
except (FileNotFoundError, ValueError, IndexError) as e:
    print(f"Error: {e}")