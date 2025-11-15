import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.arange(0.0, 10.0, 0.02)

# Calculate y values (sine(3*pi*x))
y = np.sin(3 * np.pi * x)

# Create the figure and axes object
fig, ax = plt.subplots(figsize=(4, 4))  # 4x4 inch figure

# Plot the line
ax.plot(x, y)

# Set x-axis limits
ax.set_xlim([-2, 10])

# Set y-axis limits
ax.set_ylim([-6, 6])



plt.savefig('/app/agents/text2chart/v1/save/matplotbench_easy/9/v1.png'); plt.close()