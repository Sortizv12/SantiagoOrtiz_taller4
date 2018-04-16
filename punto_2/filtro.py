import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import convolve2d

print "Ingrese el nombre del archivo de la imagen en comillas"
imagen=input()
print "Ingrese bajo si quiere un filtro pasabajas o alto si quiere uno pasaaltas en comillas"
tipo_filtro=input()

imge=plt.imread("gat.png")
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
	(fils,cols)=np.shape(matr)
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

def shifting(x):
	mitx=np.shape(x)[0]
	mity=np.shape(x)[1]
	a,b,c,d=x[:mitx, :mity], x[mitx:, :mity], x[:mitx, mity:], x[mitx:, mity:]
	return np.concatenate((np.concatenate((d,c)),np.concatenate((b,a))),axis=1)
	
def altas(matriz):
	f=np.shape(matriz)[0]
	c=np.shape(matriz)[1]
	for i in range(f):
		for j in range(c):
			if np.sqrt((i-f)**2+(j-c)**2)-10<0.07*f:
				matriz[i][j]=1*matriz[i][j]
			elif (np.sqrt((i-f)**2+(j-c)**2))+10>0.07*f:
				matriz[i][j]=0
			else:
				matriz[i][j]=0.5*(1-np.sin(np.pi*(i-0.07*f)/(20)))
	return matriz

def bajas(matriz):
	f=np.shape(matriz)[0]
	c=np.shape(matriz)[1]
	for i in range(f):
		for j in range(c):
			if np.sqrt((i-f)**2+(j-c)**2)-10<0.07*f:
				matriz[i][j]=1*matriz[i][j]
			elif (np.sqrt((i-f)**2+(j-c)**2))+10>0.07*f:
				matriz[i][j]=0
			else:
				matriz[i][j]=0.5*(1-np.sin(np.pi*(i-0.07*f)/(20)))
	return matriz
if tipo_filtro=="alto":
	red1=inversa_2D(shifting(altas(shifting(fourier_2D(tr_rojo))))).real
	gre1=inversa_2D(shifting(altas(shifting(fourier_2D(tr_verde))))).real
	blu1=inversa_2D(shifting(altas(shifting(fourier_2D(tr_azul))))).real

	filtrada=convertir_a_imagen(red1,gre1,blu1,imge)
	plt.imshow(filtrada)
	plt.savefig("altas.png")

elif tipo_filtro=="bajo":
	red1=inversa_2D(shifting(bajas(shifting(fourier_2D(tr_rojo))))).real
	gre1=inversa_2D(shifting(bajas(shifting(fourier_2D(tr_verde))))).real
	blu1=inversa_2D(shifting(bajas(shifting(fourier_2D(tr_azul))))).real

	filtrada=convertir_a_imagen(red1,gre1,blu1,imge)
	plt.imshow(filtrada)
	plt.savefig("bajas.png")

	
	
	
