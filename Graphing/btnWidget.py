import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.widgets import Button

#making sine waves as demonstration data
fre = np.arange(2,20,3)
t = np.arange(0.0,1.0,0.001)
s = np.sin(2*np.pi*fre[0]*t)

#making subplot for button widget
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
l, = plt.plot(t,s)

class btn:
    indx = 0

    def next(self, event):
        self.indx +=1
        i = self.indx % len(fre)
        ydata = np.sin(2*np.pi*fre[i]*t)
        l.set_ydata(ydata)
        plt.draw()

g = btn()
btnExpand = Button(plt.axes([0.7,0.01,0.2,0.125]),"Expand", color = "darkturquoise")
btnExpand.on_clicked(g.next)


plt.show()