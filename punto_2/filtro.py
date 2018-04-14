import numpy as np
import matplotlib.pyplot as plt


def fourier(arreglo,frec):
	for n in range(0,Np):
		zsum=complex(0.0,0.0)
		for k in range(0,N):
			expon=complex(0,2.0*np.pi*k*(n/N))
			zsum+= arreglo[k]*np.exp(-expon)
		frec[n]=zsum/np.sqrt(2.0*np.pi)






o r n i n r a n g e ( 0 , Np ) :
zsum = complex ( 0 . 0 , 0 . 0 )
# r e a l and imag p a r t s = z e r o
f o r k i n r a n g e ( 0 , N) :
# l o o p f o r sums
z e x p o = complex ( 0 , t w o p i ∗k∗n / N)
# complex e x p o n e n t
zsum += s i g n a l [ k ]∗ exp(−z e x p o )
# Fourier transform core
d f t z [ n ] = zsum ∗ s q 2 p i
# factor
i f d f t z [ n ] . imag ! = 0 :
# p l o t i f n o t t o o s m a l l imag
i m p a r t . p l o t ( p o s = ( n , d f t z [ n ] . imag ) )
# plot bars
	
