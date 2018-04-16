import numpy as np
import matplotlib.pyplot as plt
def polin_interp(lin, xx, yy):
	p=[]
	for i in range(len(xx)):
		mult=1
		for j in range(len(xx)):
			if i!=j:		
				mult=mult*((lin-xx[j])/(xx[i]-xx[j]))
		p.append(mult*yy[i])
	return sum(p)


