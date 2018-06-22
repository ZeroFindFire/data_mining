#coding=utf-8

import dt 
import k_mean
import numpy as np 

def tst_avg(dts):
	sx = sum([dt[0] for dt in dts])
	sy = sum([dt[1] for dt in dts])
	l = len(dts)
	if l == 0:
		return 0,0
	else:
		l = 1.0/l
	return sx*l, sy*l


tst_dst =lambda a,b:np.abs(a[0]-b[0]) + np.abs(a[1]-b[1])

dts = dt.point(0,0,300,10,10) + dt.point(20,20,300,10,10) + dt.point(-20,60,300,10,10)
def test(count = 300, k = 3):
	global dts 
	fc_done=k_mean.DefaultDone(count)

	rst=k_mean.k_means(dts,k,fc_done)

	avgs=rst[0]
	dtss = rst[1]
	print len(avgs),len(dtss)
	for i in xrange(len(avgs)):
		avg=avgs[i]
		dt.draw([avg],c=dt.color(i),s =30)
		tdts = dtss[i]
		if len(tdts)==0:
			continue 
		dt.draw(tdts, c = dt.color(i), s = 1)
	dt.show()
"""

python
from dtm import demo
demo.test()

"""