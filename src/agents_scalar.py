# src/agents_scalar.py
import numpy as np
from src.mpc_scalar import (
    init_mpc_with_armax_scalar, update_mpc_parameters_scalar,
    solve_mpc, get_values_scalar
)

class RB: # Rule-Based controller
    """Simple rule-based controller.
    Turns heating on if temperature < T_min.
    Turns heating off if temperature >= T_max.
    """
    def __init__(self, n_zones, T_min, T_max):
        # TODO: Enable arrays T_min and T_max (e.g. at night). Require timestamp
        self.zones = n_zones
        self.T_min, self.T_max = T_min, T_max
        self.heating_on = np.ones(n_zones, dtype=bool)

    def rule_based_control_by_zone(self, xt):
        xt = np.asarray(xt)
        # turn on where currently off AND below min
        to_on = (~self.heating_on) & (xt < self.T_min)
        # turn off where currently on  AND above max
        to_off = self.heating_on  & (xt >= self.T_max)
        self.heating_on[to_on]  = True
        self.heating_on[to_off] = False
        return self.heating_on.astype(int) # ut

    def predict(self, observations):
        xt = observations.x[-1]  # Get the current state
        ut = self.rule_based_control_by_zone(xt)
        return ut


class MPCScalar:
    """Single-zone MPC using averaged (scalar) ARMAX."""
    def __init__(self, avg_config, target_temperature, T_min, T_max,
                 history_length, horizon_length, objective='tracking'):
        self.mpc = init_mpc_with_armax_scalar(
            avg_config, target_temperature, T_min, T_max,
            history_length, horizon_length, objective
        )
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.results = {
            'temperature': [], 'control_action': [], 'ambient_temperature': [],
            'solar_irradiance': [], 'solving_time': [], 'slack': []
        }

    def predict(self, observations, solver_name='gurobi'):
        # Average zone histories to scalar !!!
        x_hist = observations.x.mean(axis=1)  # shape: (H+1,)
        u_hist = observations.u.mean(axis=1)  # shape: (H+1,)
        T_amb  = observations.a               # (H+Hh,)
        Q_irr  = observations.s               # (H+Hh, n_solar)

        update_mpc_parameters_scalar(self.mpc, x_hist, u_hist, T_amb, Q_irr, self.history_length)
        self.solving_time = solve_mpc(self.mpc, solver_name)
        u0 = self.mpc.u[0, 0].value #only returns a scalar optimal valve action!
        return [u0]  # keep as a list form for Env.step()

    def get_values(self):
        return get_values_scalar(self.mpc)

    def save_episode(self):
        x, u, a, s, slack = self.get_values()
        self.results['temperature'].append(x)
        self.results['control_action'].append(u)
        self.results['ambient_temperature'].append(a)
        self.results['solar_irradiance'].append(s)
        self.results['solving_time'].append(self.solving_time)
        self.results['slack'].append(slack)