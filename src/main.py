import numpy as np
from gmm import GMM
from plot_gmm import draw2dgmm
from test_func import noisy_cosine
import pylab as pl

x,y = noisy_cosine()
data = np.vstack([x,y]).transpose()
pl.scatter(data[:,0],data[:,1])
gmm = GMM(dim = 2, ncomps = 2, data = data, method = "kmeans")
draw2dgmm(gmm)

nx = np.arange(0,2 * np.pi, 0.1)
ny = []
for i in nx:
    ngmm = gmm.condition([0],[i])
    ny.append(ngmm.mean()) 
pl.show()
# pl.plot(nx,ny,color='red')
