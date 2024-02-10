import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7) #memorized path for the "randomness"
sampleData = {
    "a":    np.arange(50),
    "b":    np.random.randint(0,50,50),
    "c":    np.random.randn(50)
}
sampleData["d"] = sampleData["a"] + 10 * np.random.randn(50)
sampleData["c"] = np.abs(sampleData["c"]) * 100

fig, ax = plt.subplots(figsize = (5, 3), layout = "constrained") #figsize adjusts the graph while layout "constrained" adjusts spacing automatically
ax.scatter("a", "d", c = "b", s = "c",data = sampleData)
ax.set_xlabel("Entry A")
ax.set_ylabel("Entry D")
plt.show()