import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.lib.npyio import loadtxt

# load_arr = loadtxt('C:/thesis_code/Github/data/numpy_arrays/np_array')
# np_array = load_arr.reshape(load_arr[0], load_arr[1] // )

data = np.load('C:/thesis_code/Github/data/numpy_arrays/np_array1.npy')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z, x, y = data.nonzero()
ax.scatter(x, y, z, zdir='z')
plt.savefig('C:/thesis_code/Github/plotting/plots/test.png')

# legend eller colorbar med hvad farverne indikerer
# Find ud af hvordan jeg kan ændre akserne
# fjerne "nullerne"
