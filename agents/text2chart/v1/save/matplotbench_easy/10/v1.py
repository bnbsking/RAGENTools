import matplotlib.pyplot as plt
import numpy as np

# Generate the numerical sequence
x = np.arange(0.0, 3.0, 0.02)

# Calculate the data for the three lines
y1 = x**2  # Square of the sequence
y2 = np.cos(3 * np.pi * x)  # Cosine of 3*pi times the sequence
y3 = y1 * y2  # Product of the square and the cosine

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
plt.plot(x, y1, label='square')
plt.plot(x, y2, label='oscillatory')
plt.plot(x, y3, label='damped')

# Add labels and title
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Damped oscillation')

# Add the legend
plt.legend()

# Display the plot
plt.grid(True)  # Add grid for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping

plt.savefig('/app/agents/text2chart/v1/save/matplotbench_easy/10/v1.png'); plt.close()