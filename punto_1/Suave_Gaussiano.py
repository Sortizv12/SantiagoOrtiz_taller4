import numpy as np 
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import convolve2d

ksize = 30
img=plt.imread("uniandes.png")
[X,Y]=[np.array(range(ksize)[1:]),np.array(range(ksize)[1:])]
def gaussiana(s,k,X,Y):
	kernel=np.zeros((ksize-1,ksize-1))
	for i in range(ksize-1):
		for j in range(ksize-1):
			kernel[i][j]=(1/(2*np.pi*(s**2)))*np.exp(-((X[i]-k)**2 + (Y[j]-k)**2)/(2*(s**2)))
	return kernel/np.linalg.norm(kernel)



# padded fourier transform, with the same shape as the image
kernel_ft = fftpack.fft2(gaussiana(0.05,ksize/2,X,Y), shape=img.shape[:2], axes=(0, 1))

# convolve
img_ft = fftpack.fft2(img, axes=(0, 1))
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real


plt.figure()
#plt.imshow(gaussiana(3,ksize/2,X,Y),cmap='Greys_r')
plt.imshow(img)
plt.show()
plt.close()
plt.imshow(img2)
plt.show()




