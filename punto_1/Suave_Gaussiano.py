import numpy as np 
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import convolve2d

size = 30
img=plt.imread("gat.png")

def gaussiana(s,k,X,Y,ksize):
	kernel=np.zeros((ksize,ksize))
	for i in range(ksize):
		for j in range(ksize):
			kernel[i][j]=(1/(2*np.pi*(s**2)))*np.exp(-((X[i]-k)**2 + (Y[j]-k)**2)/(2*(s**2)))
	return kernel
fils=np.shape(img)[0]
cols=np.shape(img)[1]
[X,Y]=[np.array(range(fils)),np.array(range(cols))]
mrojo=[]
mverde=[]
mazul=[]
def rellenar(kernel0,img):
	kernel=np.copy(img)
	for i in range(fils):
		for j in range(cols):
			if(i<size and j<size):
				kernel[i][j]=kernel0[i][j]
			else:
				kernel[i][j]=0	
	return kernel
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
tfilroj=[]
tfilver=[]
tfilazu=[]
tfilkernel=[]
for i in range(fils):
	tfilroj.append(fourier(mrojo[i],fils))
	tfilver.append(fourier(mverde[i],fils))
	tfilazu.append(fourier(mazul[i],fils))
	tfilkernel.append(fourier(rellenar(gaussiana(2,size/2,X,Y,size),mrojo)[i],fils))
	
tfilro=np.array(tfilroj)
tfilve=np.array(tfilver)
tfilaz=np.array(tfilazu)
tfilkern=np.array(tfilkernel)
tcolroj=[]
tcolver=[]
tcolazu=[]
tcolkernel=[]
for j in range(cols):
	tcolroj.append(fourier(np.transpose(tfilro)[j],cols))
	tcolver.append(fourier(np.transpose(tfilve)[j],cols))
	tcolazu.append(fourier(np.transpose(tfilaz)[j],cols))
	tcolkernel.append(fourier(np.transpose(rellenar(gaussiana(2,size/2,X,Y,size),mrojo))[j],cols))
tr_rojo=np.transpose(np.array(tcolroj))
tr_verde=np.transpose(np.array(tcolroj))
tr_azul=np.transpose(np.array(tcolroj))
tr_kern=np.transpose(np.array(tcolkernel))

for i in range(fils):
	for j in range(cols):
		tr_rojo[i][j]=tr_rojo[i][j]*tr_kern[i][j]
		tr_verde[i][j]=tr_rojo[i][j]*tr_kern[i][j]
		tr_azul[i][j]=tr_rojo[i][j]*tr_kern[i][j]

for i in range(fils):
	tr_rojo[i]=inversa(tr_rojo[i],fils)
	tr_verde[i]=inversa(tr_verde[i],fils)
	tr_azul[i]=inversa(tr_azul[i],fils)

for j in range(cols):
	tr_rojo[i]=inversa(np.transpose(tr_rojo)[j],cols)
	tr_verde[i]=inversa(np.transpose(tr_verde)[j],cols)
	tr_azul[i]=inversa(np.transpose(tr_azul)[j],cols)


for i in range(fils):
	for j in range(cols):
		img[i][j][0]=tr_rojo[i][j]
		img[i][j][1]=tr_verde[i][j]
		img[i][j][2]=tr_azul[i][j]


# padded fourier transform, with the same shape as the image
#kernel_ft = fftpack.fft2(gaussiana(0.05,ksize/2,X,Y), shape=img.shape[:2], axes=(0, 1))

# convolve
#img_ft = fftpack.fft2(img, axes=(0, 1))
#img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
#img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real


#plt.figure()
#plt.imshow(gaussiana(3,ksize/2,X,Y),cmap='Greys_r')

plt.imshow(img)
plt.show()




