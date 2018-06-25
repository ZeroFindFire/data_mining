#coding=utf-8

import dt 
import cluster
import numpy as np 
dts = None
#init()
def entropy(ls):
	s = sum(ls)
	s = 1.0 / s 
	fs = [s * l for l in ls]
	
	import numpy
	rst = - sum(f * numpy.log(f) for f in fs)
	return rst

def rotate(x,y,angle):
	angle = angle * np.pi / 180
	xx = x * np.cos(angle)
	xy = x * np.sin(angle)
	yx = - y * np.sin(angle)
	yy = y * np.cos(angle)
	return [xx + yx, xy + yy]
def init():
	global dts 
	dts = dt.circle(0,0,30,10,10) + dt.circle(20,20,30,10,10) + dt.circle(-20,60,30,10,10)
	dts = dt.circle(0,0,30,5,5) + dt.ring(0,0,20,200,5)
	#dts = [[np.abs(tdt[0]),np.abs(tdt[1])] for tdt in dts]
	#dts = [rotate(tdt[0],tdt[1],-45) for tdt in dts]


def show():
	dt.show_pt(dts,c=dt.color(0), s= 1)

def test(count = 1300, k = 2, mx = True):
	global dts 
	
	#dts = [[tdt[0] **2,tdt[1]**2] for tdt in dts]
	#dt.show_pt(dts,c=dt.color(0), s= 1)
	fc_done=cluster.DefaultDone(count)

	rst=cluster.k_means(dts,k,fc_done,cluster.default_dst,mx)

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
	return None
	tree = cluster.cluster(dts,k, cluster.min_dsts)
	print "done tree"
	sets = cluster.tree2sets(tree)
	#sets = cluster.cut_tree(tree,k,cluster.min_cost)
	print "done cut"
	print len(sets)
	for i in xrange(len(sets)):
		tdts = sets[i]
		if len(tdts)==0:
			continue 
		dt.draw(tdts, c = dt.color(i), s = 1)
	dt.show()
	return sets
"""

python
from dtm import demo
demo.init()
rst=demo.test(mx=False),demo.test(mx=True)

"""