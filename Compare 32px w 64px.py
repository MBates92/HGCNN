import numpy as np
import matplotlib.pyplot as plt

array_32px_000 = np.loadtxt('D:/Workshop/PhD/Data/ViaLactea/CNN_Results/32px/H/avg_data/000.txt')
array_64px_000 = np.loadtxt('D:/Workshop/PhD/Data/ViaLactea/CNN_Results/64px/H/avg_data/000.txt')

plt.figure()
plt.imshow(array_32px_000)
plt.show()

plt.figure()
plt.imshow(array_64px_000)
plt.show()