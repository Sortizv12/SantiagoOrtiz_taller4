#include <iostream>
using namespace std:

double lagrange_interp(double* lin,double* xx,double* yy);

int main()
{



return 0;
}

double lagrange_interp(double* lin,double* xx, double* yy)
{
	int len=sizeof(xx)/sizeof(*xx);
	double p[];
	for(int i=0;i<len;i++)
	{
		double mult=1.0;
		for(int j=0;j<len;j++)
		{
			if(i!=j)
			{
				mult=mult*((lin-xx[j])/(xx[i]-xx[j]));
			}
		p[i]=mult*yy[i];
		}
	}
	double sum=0.0;
	for(int k=0;k<len;k++)
	{
		sum+=p[i];
	}
	

}

//def polin_interp(lin, xx, yy):
//	p=[]
//	for i in range(len(xx)):
//		mult=1
//		for j in range(len(xx)):
//			if i!=j:		
//				mult=mult*((lin-xx[j])/(xx[i]-xx[j]))
//		p.append(mult*yy[i])
//	return sum(p)
