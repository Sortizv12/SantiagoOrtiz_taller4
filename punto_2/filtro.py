import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("uniandes.png")
def fourier(arreglo,frec):
	for n in range(0,Np):
		zsum=complex(0.0,0.0)
		for k in range(0,N):
			expon=complex(0,2.0*np.pi*k*(n/N))
			zsum+= arreglo[k]*np.exp(-expon)
		frec[n]=zsum/np.sqrt(2.0*np.pi)
	return frec




	
