import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

time = 0

def explorer(x_t: np.array, u_bounds: dict, timestep: int) -> np.array:

    global time 

    u_lower = u_bounds['low']
    u_upper = u_bounds['high']

    u_0 = u_lower
    u_1 = u_lower + (u_upper-u_lower)/3
    u_2 = u_lower + 2*(u_upper-u_lower)/3
    u_3 = u_upper
    
    up_up = False
    down_down = False
    up_halfup = False
    down_halfdown = False
    up_down_up = False

    if time < 60 or time >= 300:
        up_up = True
    elif time >= 60 and time < 120:
        down_down = True
    elif time >= 120 and time < 180:
        up_halfup = True
    elif time >= 180 and time < 240:
        down_halfdown = True
    else:
        up_down_up = True
    
    if up_up == True:
        if timestep < 15:
            u_plus = u_0
        elif timestep >= 15 and timestep < 30:
            u_plus = u_1
        elif timestep >= 30 and timestep < 45:
            u_plus = u_2
        else:
            u_plus = u_3
    
    elif down_down == True:
        if timestep < 15:
            u_plus = u_3
        elif timestep >= 15 and timestep < 30:
            u_plus = u_2
        elif timestep >= 30 and timestep < 45:
            u_plus = u_1
        else:
            u_plus = u_0

    elif up_halfup == True:
        if timestep < 15:
            u_plus = np.array([u_0[0],u_0[0]])
        elif timestep >= 15 and timestep < 30:
            u_plus = np.array([u_1[0],u_1[0]])
        elif timestep >= 30 and timestep < 45:
            u_plus = np.array([u_2[0],u_2[0]])
        else:
            u_plus = np.array([u_3[0],u_3[0]])

    elif down_halfdown == True:
        if timestep < 15:
            u_plus = np.array([u_3[0],u_3[0]])
        elif timestep >= 15 and timestep < 30:
            u_plus = np.array([u_2[0],u_2[0]])
        elif timestep >= 30 and timestep < 45:
            u_plus = np.array([u_1[0],u_1[0]])
        else:
            u_plus = np.array([u_0[0],u_0[0]])

    elif up_down_up == True:
        if timestep < 15:
            u_plus = np.array([u_3[0],u_0[1]])
        elif timestep >= 15 and timestep < 30:
            u_plus = np.array([u_1[0],u_1[0]])
        elif timestep >= 30 and timestep < 45:
            u_plus = np.array([u_3[0],u_3[0]])
        else:
            u_plus = np.array([u_3[0],u_0[1]])

    time += 1

    return u_plus

def model_trainer(data: np.array, env: callable) -> callable:
    data_states, data_controls = data
    
    # Select specific states (indices 1 and 8) and normalize
    selected_states = data_states[:, [1, 8], :]
    o_low, o_high = env.env_params['o_space']['low'][[1, 8]], env.env_params['o_space']['high'][[1, 8]]
    a_low, a_high = env.env_params['a_space']['low'], env.env_params['a_space']['high']
    selected_states_norm = (selected_states - o_low.reshape(1, -1, 1)) / (o_high.reshape(1, -1, 1) - o_low.reshape(1, -1, 1)) * 2 - 1
    data_controls_norm = (data_controls - a_low.reshape(1, -1, 1)) / (a_high.reshape(1, -1, 1) - a_low.reshape(1, -1, 1)) * 2 - 1

    # Extract dimensions
    reps, states, n_steps = selected_states_norm.shape
    _, controls, _ = data_controls_norm.shape
    
    # Prepare X (current state + control) and y (change in state)
    X_states = selected_states_norm[:, :, :-1].reshape(-1, states)  # state[t]
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)  # control[t]
    X = np.hstack([X_states, X_controls])  # Inputs: [state[t], control[t]]
    
    # Compute changes in states as output: Δstate[t] = state[t+1] - state[t]
    y_diff = selected_states_norm[:, :, 1:] - selected_states_norm[:, :, :-1]  # Shape: (reps, states, n_steps-1)
    y = y_diff.reshape(-1, states)  # Shape: (reps * (n_steps-1), states)

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train MultiTaskLassoCV on full data to predict changes
    model = MultiTaskLassoCV(alphas=np.logspace(-4, 1, 50), cv=tscv, random_state=42)
    model.fit(X_scaled, y)

    # Define predictor for changes
    def next_state_predictor(x, u):
        """
        Predicts the next state given current state (x) and control (u).
        Model predicts Δstate[t], so next state = x + Δstate[t].
        """
        X_in = np.hstack([x, u]).reshape(1, -1)
        X_in_scaled = scaler.transform(X_in)
        delta_state = model.predict(X_in_scaled)  # Predicted change: Δstate[t]
        next_state = x + delta_state.flatten()  # Next state = current state + change
        return next_state

    return next_state_predictor

def controller(x: np.array, f: callable, sp: callable, env: callable, u_prev: np.array,) -> np.array:
    controller.team_names = ['Donggyu Lee']
    controller.cids = ['01560108']

    o_space = env.env_params['o_space']
    a_space = env.env_params['a_space']
    
    Q = 25  # Increased state cost for more precision
    R = 7   # Reduced control cost for smoother control actions
    
    horizon = 2
    x_current = x[1]

    n_controls = a_space['low'].shape[0]
    u_prev = (u_prev - a_space['low']) / (a_space['high'] - a_space['low']) * 2 - 1

    def predict_next_state(current_state, control):
        current_state_norm = (current_state - o_space['low'][[1, 8]]) / (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) * 2 - 1
        prediction = f(current_state_norm, control).flatten()
        return (prediction + 1) / 2 * (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) + o_space['low'][[1, 8]]

    def objective(u_sequence):
        cost = 0
        x_pred = x_current
        for i in range(horizon):
            u_current = u_sequence[i*n_controls:(i+1)*n_controls]
            error = x_pred - sp
            cost += np.sum(error**2) * Q
            cost += np.sum((u_current - u_prev)**2) * R
            x_pred = predict_next_state(x_pred, u_current)
            u_current_denorm = (u_current + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']
        return cost

    u_init = np.ones(horizon * n_controls)
    bounds = [(-1, 1)] * (horizon * n_controls)

    result = minimize(objective, u_init, method='SLSQP', bounds=bounds)

    optimal_control = result.x[:n_controls]

    return (optimal_control + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']