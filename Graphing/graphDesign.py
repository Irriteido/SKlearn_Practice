import matplotlib.pyplot as plt
import numpy as np

sample_data1 = np.random.randn(4,100)
sample_data2 = np.random.randn(4,100)
fig, ax = plt.subplots(figsize = (5,3))

ax.plot(np.arange(len(sample_data1)), sample_data1,
         color = "blue", label="sample data 1", linestyle = "--")
ax.plot(np.arange(len(sample_data2)), sample_data2,
        color = "red", label = "sample data 2", linestyle= "-.")

plt.show()