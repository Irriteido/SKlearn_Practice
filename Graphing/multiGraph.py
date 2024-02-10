import matplotlib.pyplot as plt
import numpy as np

sample_x = np.linspace(0,2,100)

plt.figure(figsize=(5,3), layout="constrained")
plt.plot(sample_x, sample_x, label = "linear")
plt.plot(sample_x, sample_x**2, label = "quadratic")
plt.plot(sample_x, sample_x**3, label = "cubic")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()