import numpy as np 
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import convolve2d

size = 60
imge=plt.imread("gat.png")

def gaussiana(s,k,X,Y,ksize):
	kernel=np.zeros((ksize,ksize),dtype=complex)
	for i in range(ksize):
		for j in range(ksize):
			kernel[i][j]=(1/(2*np.pi*(s**2)))*np.exp(-((X[i]-k)**2 + (Y[j]-k)**2)/(2*(s**2)))
	return kernel
fils=np.shape(imge)[0]
cols=np.shape(imge)[1]
[X,Y]=[np.array(range(fils)),np.array(range(cols))]

def rellenar(kernel0,img):
	kernel=np.zeros(np.shape(img),dtype=complex)
	for i in range(np.shape(img)[0]):
		for j in range(np.shape(img)[1]):
			if(i<len(kernel0) and j<len(kernel0)):
				kernel[i][j]=kernel0[i][j]
			else:
				kernel[i][j]=0	
	return kernel
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
def convertir_a_imagen(matr,matv,mata,img):
	
	for i in range(fils):
		for j in range(cols):
			img[i][j][0]=tr_rojo[i][j]
			img[i][j][1]=tr_verde[i][j]
			img[i][j][2]=tr_azul[i][j]
	return img
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

#plt.imshow(fourier_2D(rellenar(kernel,tr_rojo)).real)
#plt.show()
conv_rojo=fourier_2D(tr_rojo)*fourier_2D(rellenar(gaussiana(4,size/2,X,Y,size),tr_rojo))
conv_verde=fourier_2D(tr_verde)*fourier_2D(rellenar(gaussiana(4,size/2,X,Y,size),tr_verde))
conv_azul=fourier_2D(tr_azul)*fourier_2D(rellenar(gaussiana(4,size/2,X,Y,size),tr_azul))	

filtrada=convertir_a_imagen(inversa_2D(conv_rojo).real,inversa_2D(conv_verde).real,inversa_2D(conv_azul).real,imge)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(filtrada)
plt.subplot(1,2,2)
plt.imshow(imge)
#plt.subplot(1,3,3)
#plt.imshow(rellenar(gaussiana(4,size/2,X,Y,size),tr_azul))
plt.savefig("suave.png")
plt.show()




