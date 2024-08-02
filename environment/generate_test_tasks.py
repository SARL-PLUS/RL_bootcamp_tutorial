import os

import numpy as np
from matplotlib import pyplot as plt, colors

from environment.environment_awake_steering import MamlHelpers

# Define file location and name
verification_tasks_loc = 'tasks'
filename = 'verification_tasks.pkl'  # Adding .pkl extension for clarity

# Create an instance of MamlHelpers and sample tasks
nr_tasks = 5
tasks_new = [MamlHelpers().get_origin_task(idx=0)] + [{'id': task['id'] + 1, **task}
                                                      for task in MamlHelpers().sample_tasks(nr_tasks)]
tasks = []
max_variation = 20
nr_tasks = 5  # Assuming nr_tasks is defined

while len(tasks) < nr_tasks:
    sampled_task = MamlHelpers().sample_tasks(num_tasks=1)[0]  # Assuming sample_tasks returns a list of tasks
    task_goal_values = sampled_task['goal'][0]

    max_value = np.max(task_goal_values)
    min_value = np.min(task_goal_values)

    if max_value < max_variation and abs(min_value) < max_variation:
        tasks.append(sampled_task)

tasks_new = [MamlHelpers().get_origin_task(idx=0)] + [{'id': task['id'] + 1, **task}
                                                      for task in tasks]
# print(tasks_new)

num_tasks = len(tasks_new)

# Determine the number of rows and columns for the subplot grid
cols = int(np.ceil(np.sqrt(num_tasks)))
rows = int(np.ceil(num_tasks / cols))

fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Find the global min and max across all tasks for normalization
all_values = [task['goal'][0] for task in tasks_new]
global_min = min(map(np.min, all_values))
global_max = max(map(np.max, all_values))

# Define a common normalization
norm = colors.Normalize(vmin=global_min, vmax=global_max)

for nr, task in enumerate(tasks_new):
    a = task['goal'][0]
    im = axs[nr].matshow(a, aspect='equal', norm=norm)
    axs[nr].set_title(f"Task {nr+1}")

# Turn off axes for any empty subplots
for nr in range(num_tasks, len(axs)):
    axs[nr].axis('off')

# Add a global colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(verification_tasks_loc + '/overview_responses_tasks_valid.pdf')
plt.show()






# Construct the full file path
full_path = os.path.join(verification_tasks_loc, filename)

# Save the tasks using pickle
# with open(full_path, "wb") as fp:
#     pickle.dump(tasks_new, fp)


# import yaml
#
# config_path = 'tasks/maml/awake.yaml'
# kwargs = {'verification_tasks_loc': full_path}
#
# # Load the configuration from the YAML file and update 'env-kwargs'
# with open(config_path, "r") as file:
#     config = yaml.safe_load(file)
#     config.setdefault("env-kwargs", {}).update(kwargs)
#
# # Optional: If you need to write the updated config back to the file
# with open(config_path, "w") as file:
#     yaml.dump(config, file)



