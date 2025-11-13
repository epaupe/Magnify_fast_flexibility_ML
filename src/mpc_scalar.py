# src/mpc_scalar.py
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def armax_scalar_model(mpc, t, lags, A, B, c, D, b):
    if t < 1:
        return pyo.Constraint.Skip
    expr = b
    for lag in lags:
        expr += A[lag] * mpc.x[t - lag, 0]
        expr += B[lag] * mpc.u[t - lag, 0]
        expr += c[lag] * mpc.a[t - lag]
        # solar bins
        for j in mpc.solar_terms:
            expr += D[lag][j] * mpc.s[t - lag, j]
    return mpc.x[t, 0] == expr


def init_mpc_with_armax_scalar(avg_config, target_temperature, T_min, T_max,
                               history_length, horizon_length, objective='tracking'):
    """
    Single-zone (scalar) ARMAX MPC using averaged coefficients.
    """
    lags = avg_config["dynamics_lags_list"]
    A    = avg_config["avg_params"]["A"]
    B    = avg_config["avg_params"]["B"]
    c    = avg_config["avg_params"]["c"]
    D    = avg_config["avg_params"]["D"]
    b    = avg_config["avg_params"]["b"]
    n_s  = avg_config["solar_terms"]

    mpc = pyo.ConcreteModel()
    mpc.zone_range = pyo.RangeSet(0, 0)  # single zone index 0
    mpc.time_horizon       = pyo.RangeSet(0, horizon_length)
    mpc.time_horizon_input = pyo.RangeSet(0, horizon_length - 1)
    mpc.time_state         = pyo.RangeSet(-history_length, horizon_length)
    mpc.time_input         = pyo.RangeSet(-history_length, horizon_length - 1)
    mpc.solar_terms        = pyo.RangeSet(0, n_s - 1)

    # Variables
    mpc.u     = pyo.Var(mpc.time_input, mpc.zone_range, bounds=(0, 1))
    mpc.x     = pyo.Var(mpc.time_state, mpc.zone_range)
    mpc.slack = pyo.Var(mpc.time_horizon, mpc.zone_range, within=pyo.NonNegativeReals)

    # External inputs / params
    mpc.a       = pyo.Param(mpc.time_input, mutable=True, default=0.0)  # ambient
    mpc.s       = pyo.Param(mpc.time_input, mpc.solar_terms, mutable=True, default=0.0)
    mpc.T_min   = pyo.Param(mpc.time_horizon, mpc.zone_range,
                            initialize=lambda m, t, r: T_min[t, r], mutable=False)
    mpc.T_max   = pyo.Param(mpc.time_horizon, mpc.zone_range,
                            initialize=lambda m, t, r: T_max[t, r], mutable=False)
    mpc.T_target = pyo.Param(mpc.time_horizon, mpc.zone_range,
                             default=target_temperature, mutable=False)

    # Constraints
    mpc.dynamics = pyo.Constraint(
        mpc.time_horizon, mpc.zone_range,
        rule=lambda M, t, r: armax_scalar_model(M, t, lags, A, B, c, D, b)
    )
    mpc.bound_min = pyo.Constraint(
        mpc.time_horizon, mpc.zone_range,
        rule=lambda M, t, r: M.x[t, r] >= M.T_min[t, r] - M.slack[t, r]
    )
    mpc.bound_max = pyo.Constraint(
        mpc.time_horizon, mpc.zone_range,
        rule=lambda M, t, r: M.x[t, r] <= M.T_max[t, r] + M.slack[t, r]
    )

    # Objective
    alpha_factor = 3 / (horizon_length)
    if objective == 'tracking':
        mpc.objective = pyo.Objective(
            expr=sum(
                sum(mpc.slack[t, 0] for t in mpc.time_horizon) +
                sum((mpc.x[t, 0] - mpc.T_target[t, 0]) ** 2 for t in mpc.time_horizon)
            ), sense=pyo.minimize
        )
    elif objective == 'upper_bound':
        mpc.objective = pyo.Objective(
            expr=100 * sum(mpc.slack[l, 0] for l in mpc.time_horizon)
                 - sum(mpc.u[t, 0] * pyo.exp(-alpha_factor * t) for t in mpc.time_horizon_input),
            sense=pyo.minimize
        )
    elif objective == 'lower_bound':
        mpc.objective = pyo.Objective(
            expr=100 * sum(mpc.slack[l, 0] for l in mpc.time_horizon)
                 + sum(mpc.u[t, 0] * pyo.exp(-alpha_factor * t) for t in mpc.time_horizon_input),
            sense=pyo.minimize
        )
    else:
        raise ValueError("Invalid objective: 'tracking' | 'upper_bound' | 'lower_bound'")

    return mpc


def update_mpc_parameters_scalar(mpc, history_x_avg, history_u_avg, T_amb, Q_irr, history_length):
    """
    Fix history and update exogenous series for scalar model.
    history_x_avg, history_u_avg shape: (history_length+1,) and (history_length+1,)
    T_amb: (history_length + horizon_length,)
    Q_irr: (history_length + horizon_length, n_solar)
    """
    # Fix current state at t=0
    mpc.x[0, 0].fix(history_x_avg[-1])

    # Fix historical values for t<0
    for i, t in enumerate(range(-history_length, 0)):
        mpc.u[t, 0].fix(history_u_avg[i])
        mpc.x[t, 0].fix(history_x_avg[i])

    # Unfix future values
    for t in mpc.time_horizon:
        mpc.slack[t, 0].unfix()
        if t > 0:
            mpc.x[t, 0].unfix()
        if t < len(mpc.time_horizon) - 1:
            mpc.u[t, 0].unfix()

    # Update weather
    for i, t in enumerate(mpc.time_input):
        mpc.a[t] = float(T_amb[i])
        for j in mpc.solar_terms:
            mpc.s[t, j] = float(Q_irr[i, j])


def solve_mpc(mpc, solver_name='gurobi'):
    results = SolverFactory(solver_name).solve(mpc, tee=False)
    return results.solver.wallclock_time


def get_values_scalar(mpc):
    x = {t: {0: pyo.value(mpc.x[t, 0])} for t in mpc.time_state}
    u = {t: {0: pyo.value(mpc.u[t, 0])} for t in mpc.time_input}
    a = {t: pyo.value(mpc.a[t]) for t in mpc.time_input}
    s = {t: {j: pyo.value(mpc.s[t, j]) for j in mpc.solar_terms} for t in mpc.time_input}
    slack = {t: {0: pyo.value(mpc.slack[t, 0])} for t in mpc.time_horizon}
    return x, u, a, s, slack
