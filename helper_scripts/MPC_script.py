"""
Simon Hirlaender
This script optimizes control inputs for a model predictive control (MPC) system using the AwakeSteering environment.
The primary tasks include:
1. Defining a root mean square (RMS) function to measure deviations.
2. Predicting control actions by minimizing a cost function subject to system dynamics and constraints.
3. Visualizing results of state and control values over time, and analyzing performance across different tolerance settings in a parallelized execution environment.

Dependencies:
- autograd.numpy for automatic differentiation.
- scipy.optimize for numerical optimization.
- matplotlib for plotting results.
- concurrent.futures for parallel execution.
"""

import time
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import concurrent.futures

from environment_awake_steering import AwakeSteering, load_prefdefined_task


def rms(x):
    return np.sqrt(np.mean((x ** 2)))


def predict_actions(initial_state, horizon_length, response_matrix, threshold, tol=1e-8, disp=False):
    dimension = response_matrix.shape[0]

    def rms(x):
        return np.sqrt(np.mean((x ** 2)))

    a = np.eye(dimension)  # System dynamics matrix (identity for simplicity)

    # Initial guess for optimization (for 5D states and controls)
    z_initial = np.zeros(dimension * (horizon_length + 1) + dimension * horizon_length)
    # Set initial state value to value of system
    z_initial[:dimension] = initial_state

    def step_function(x, threshold):
        return 1 if rms(x) > threshold else 0.0
        # return 1

    def special_cost_function(x):
        return rms(x)

    # Define the objective function using autograd's numpy
    def objective(z):
        x = z[:dimension * (horizon_length + 1)].reshape((horizon_length + 1, dimension))
        cost = 0
        for i, x_vector in enumerate(x):
            cost += special_cost_function(x_vector)
        return cost

    # Define constraints (dynamics as equality constraints)
    def create_constraints():
        constraints = [
            {'type': 'eq', 'fun': lambda z: (z[:dimension] - initial_state).flatten()}]  # Start condition is flattened
        for i in range(horizon_length):
            constraints.append({
                'type': 'eq',
                'fun': (lambda z, k=i: (z[dimension * (k + 1):dimension * (k + 2)] -
                                        (step_function(rms(z[dimension * k:dimension * (k + 1)]), threshold) *
                                         (a @ z[dimension * k:dimension * (k + 1)] + response_matrix @
                                          z[dimension * (horizon_length + 1) + dimension * k:dimension
                                                                                             * (
                                                                                                     horizon_length + 1) + dimension * (
                                                                                                     k + 1)]))).flatten())
            })
        return constraints

    # Define bounds for the variables
    control_bounds = 1
    bounds = [(None, None)] * dimension * (horizon_length + 1) + [
        (-control_bounds, control_bounds)] * dimension * horizon_length  # No bounds on states, [-1, 1] on controls

    # Start timing for total execution
    start_total = time.time()
    costs = []
    rms_run = []  # List to store RMS values for each run

    def callback(z):
        current_x = z[:dimension * (horizon_length + 1)].reshape((horizon_length + 1, dimension))
        current_rms = [rms(x_vector) for x_vector in current_x]
        rms_run.append(current_rms)
        costs.append(objective(z))

    start = time.time()
    result = minimize(
        objective,
        z_initial,
        constraints=create_constraints(),
        bounds=bounds,
        method='SLSQP',
        options={'disp': disp, 'maxiter': 1000},
        tol=tol,
        callback=callback
    )

    end = time.time()
    time_run = end - start
    #
    rms_final = np.array(
        [special_cost_function(result.x[j * dimension:(j + 1) * dimension]) for j in range(horizon_length + 1)])
    final_result = result.x  # Take the last result (or any specific iteration you are interested in)
    state_final = final_result[:dimension * (horizon_length + 1)].reshape((horizon_length + 1, dimension))
    action_final = final_result[dimension * (horizon_length + 1):].reshape(horizon_length, dimension)

    return state_final, action_final, costs, rms_final, time_run


def model_predictive_control(x0, N, b, threshold, plot=False, **kwargs):
    tol = kwargs['tol'] if 'tol' in kwargs else 1e-16
    x_final, u_final, costs, rms_final, time_run = predict_actions(x0, N, b, threshold, tol=tol)
    if plot:
        plot_results(x_final, u_final, costs, time_run, threshold)
    return u_final[0]


def plot_results(x_final, u_final, costs, time_run, threshold):
    dimension = x_final.shape[1]
    # print('threshold_inner', threshold)
    # Assuming x_final is (time_steps, dimensions)
    time_steps = np.arange(x_final.shape[0])
    control_steps = np.arange(1, u_final.shape[0] + 1)

    plt.figure(figsize=(15, 15))

    # RMS Over Time
    plt.subplot(3, 1, 2)
    rms_final = [rms(x) for x in x_final]
    plt.plot(rms_final, '-o', label='RMS Over Time')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('RMS Over Time (Final Solution)')
    plt.xlabel('Time Step')
    plt.ylabel('RMS of State Vector')
    plt.legend()
    plt.grid(True)

    # States Over Time
    plt.subplot(3, 1, 1)
    for dim in range(dimension):
        plt.plot(time_steps, x_final[:, dim], label=f'State {dim + 1}')

    plt.title('States Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True)

    # Control Inputs Over Time
    plt.subplot(3, 1, 3)
    for dim in range(dimension):
        plt.plot(control_steps, u_final[:, dim], label=f'Control {dim + 1}')
    plt.title('Control Inputs Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.pause(2)

    # Cost Function Over Iterations
    plt.figure(figsize=(8, 5))
    plt.plot(costs, '-x', label='Cost Over Iterations')
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # Print timing information
    print(f"Total execution time: {time_run:.2f}s")


verification_task = load_prefdefined_task(1)
b = verification_task['goal'][0]

# b_inverse = np.linalg.inv(b)
env = AwakeSteering(task=verification_task)
b = b * env.action_scale
threshold = -env.threshold / env.state_scale
N = 4  # Number of steps

# print(threshold)

tol_values = [1e-8]  # Different tol values to test
number_of_samples = 10


# The function to run in each parallel process
def run_simulation(tol):
    results = []
    for i in range(number_of_samples):  # Modify this number based on the desired number of simulations per tol
        x0, _ = env.reset(seed=i)
        x0 /= env.state_scale
        x_final, u_final, costs, rms_final, time_run = predict_actions(x0, N, b, threshold, tol=tol)
        episode_length = sum(rms_final > threshold)
        results.append(episode_length)
        if i == number_of_samples - 1:
            plot_results(x_final, u_final, costs, time_run, threshold)
    return (tol, results)


# Main block to execute the parallel processing
if __name__ == "__main__":
    tol_results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the function over the tol_values
        future_to_tol = {executor.submit(run_simulation, tol): tol for tol in tol_values}
        for future in concurrent.futures.as_completed(future_to_tol):
            tol = future_to_tol[future]
            try:
                result = future.result()
                tol_results[result[0]] = result[1]
            except Exception as exc:
                print(f"{tol} generated an exception: {exc}")

    # Calculate statistics for each tol value
    mean_episode_lengths = {tol: np.mean(lengths) for tol, lengths in tol_results.items()}
    std_episode_lengths = {tol: np.std(lengths) for tol, lengths in tol_results.items()}

    # Display the results or plot as required
    plt.figure(figsize=(10, 5))
    plt.errorbar(tol_values, [mean_episode_lengths[tol] for tol in tol_values],
                 yerr=[std_episode_lengths[tol] for tol in tol_values], fmt='-o')
    plt.xscale('log')
    plt.title('Episode Lengths vs tol')
    plt.xlabel('tol')
    plt.ylabel('Mean Episode Length')
    plt.grid(True)
    plt.show()

    # Print statistics
    for tol in tol_values:
        print(f"tol: {tol:.1e}, Mean Episode Length: {mean_episode_lengths[tol]:.2f}, "
              f"Standard Deviation: {std_episode_lengths[tol]:.2f}")
