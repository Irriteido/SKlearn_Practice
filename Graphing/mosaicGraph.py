import matplotlib.pyplot as plt 

fig, axd = plt.subplot_mosaic(
    [["upleft", "right"],
    ["lowleft", "right"]], 
    layout="constrained"
)

axd["upleft"].set_title("upleft")
axd["lowleft"].set_title("lowleft")
axd["right"].set_title("right")
plt.show()