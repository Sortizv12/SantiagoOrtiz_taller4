import numpy as np 
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import convolve2d

print "Ingrese el nombre del archivo de la imagen entre comillas "
imagen=input()
print "Ingrese el ancho de la gaussiana entre comillas"
sigma=float(input())

size = 20
imge=plt.imread("gat.png")

def gaussiana(s,k,X,Y,ksize):
	kernel=np.zeros((ksize,ksize),dtype=complex)
	for i in range(ksize):
		for j in range(ksize):
			kernel[i][j]=np.exp(-((X[i]-k)**2 + (Y[j]-k)**2)/(2*(s**2)))*(1/(np.sqrt(2*np.pi*(s**2))))
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
	fils=np.shape(matr)[0]
	cols=np.shape(matr)[1]
	for i in range(fils):
		for j in range(cols):
			img[i][j][0]=matr[i][j]
			img[i][j][1]=matv[i][j]
			img[i][j][2]=mata[i][j]
	return img
tr_rojo=leer_imagen(imagen)[0]
tr_verde=leer_imagen(imagen)[1]
tr_azul=leer_imagen(imagen)[2]

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


transf_kernel=fourier_2D(rellenar(gaussiana(sigma*(1/np.sqrt(2*np.pi)),size/2,X,Y,size),tr_rojo))
conv_rojo=fourier_2D(tr_rojo)*transf_kernel
conv_verde=fourier_2D(tr_verde)*transf_kernel
conv_azul=fourier_2D(tr_azul)*transf_kernel	

a=inversa_2D(conv_rojo)
b=inversa_2D(conv_verde)
c=inversa_2D(conv_azul)

a=a/np.max(a)
b=b/np.max(b)
c=c/np.max(c)

filtrada=convertir_a_imagen(a.real,b.real,c.real,plt.imread(imagen))
plt.figure()
plt.imshow(filtrada)
plt.savefig("suave.png")





