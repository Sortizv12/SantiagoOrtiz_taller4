import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import convolve2d


def fourier(arreglo,Np):
	N=Np
	val=[]
	for n in range(0,Np):
		zsum=complex(0.0,0.0)
		for k in range(0,N):
			expon=complex(0,2.0*np.pi*k*(float(n)/float(N)))
			zsum+= arreglo[k]*np.exp(-expon)
		val.append(zsum) #/np.sqrt(2.0*np.pi))
	return np.array(val)

def inversa(arreglo,Np):
	N=Np
	val=[]
	for n in range(0,Np):
		zsum=complex(0.0,0.0)
		for k in range(0,N):
			expon=complex(0,2.0*np.pi*k*(float(n)/float(N)))
			zsum+= arreglo[k]*np.exp(expon)
		val.append(zsum) #/np.sqrt(2.0*np.pi))
	return np.array(val)

def leer_imagen(imagen):
	img=plt.imread(imagen)
	fils=np.shape(img)[0]
	cols=np.shape(img)[1]
	mrojo=[]
	mverde=[]
	mazul=[]
	for i in range(fils):
		frojo=[]
		fverde=[]
		fazul=[]
		for j in range(cols):
			frojo.append(img[i][j][0])
			fverde.append(img[i][j][1])
			fazul.append(img[i][j][2])
		
		mrojo.append(frojo)
		mverde.append(fverde)
		mazul.append(fazul)
	return np.array([np.array(mrojo),np.array(mverde),np.array(mazul)])

tr_rojo=leer_imagen("gat.png")[0]
tr_verde=leer_imagen("gat.png")[1]
tr_azul=leer_imagen("gat.png")[2]
print np.shape(tr_rojo)[0]
def fourier_2D(matriz):
	mat_medio1=np.zeros(np.shape(matriz),dtype=complex)
	mat_medio2=np.zeros(np.shape(matriz),dtype=complex)
	for i in range(np.shape(matriz)[0]):
		mat_medio1[i]=fourier(matriz[i],np.shape(matriz)[0])
	for j in range(np.shape(matriz)[1]):
		mat_medio2[j]=fourier(np.transpose(mat_medio1)[j],np.shape(matriz)[1])
	return np.transpose(mat_medio2)

def inversa_2D(matriz):
	mat_medio1=np.zeros(np.shape(matriz),dtype=complex)
	mat_medio2=np.zeros(np.shape(matriz),dtype=complex)
	for i in range(np.shape(matriz)[0]):
		mat_medio1[i]=inversa(matriz[i],np.shape(matriz)[0])
	for j in range(np.shape(matriz)[1]):
		mat_medio2[j]=inversa(np.transpose(mat_medio1)[j],np.shape(matriz)[1])
	return np.transpose(mat_medio2)
#print fftpack.fft2(tr_rojo)
#print "-------------------------------------------"
#print fourier_2D(tr_rojo)
plt.figure()
plt.subplot(2,1,1)
plt.imshow(tr_rojo)
#plt.imshow(fftpack.fft2(tr_rojo).real)
plt.subplot(2,1,2)
plt.imshow(inversa_2D(fourier_2D(tr_rojo)).real)
plt.show()	



	
	
	
