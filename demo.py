#coding=utf-8

import dt 
import cluster
import numpy as np 
import random
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
def init(k = 3, n = 60):
	global dts 
	dts = []
	for i in xrange(k):
		dts += dt.circle(random.randint(-30,30),random.randint(-30,30),n,random.randint(3,13),random.randint(3,13))
	#dts = dt.circle(0,0,60,10,10) + dt.circle(20,20,60,10,10) + dt.circle(-20,60,60,10,10)
	#dts = dt.circle(0,0,30,5,5) + dt.ring(0,0,20,200,5)
	#dts = [[np.abs(tdt[0]),np.abs(tdt[1])] for tdt in dts]
	#dts = [rotate(tdt[0],tdt[1],-45) for tdt in dts]
	global cut_done
	cut_done = ct_done(dts,n=0.0)

def show():
	dt.show_pt(dts,c=dt.color(0), s= 1)
ct_done = cluster.CutDone
cut_done = ct_done(dts,n=0.0)
def test(count = 1300, k = 3, mx = True, fc_done = None, nxt = False):
	global dts 
	
	#dts = [[tdt[0] **2,tdt[1]**2] for tdt in dts]
	#dt.show_pt(dts,c=dt.color(0), s= 1)
	if fc_done is None:
		fc_done=cut_done # cluster.DefaultDone(count)

	rst=cluster.k_means(dts,k,fc_done,cluster.default_dst,mx)

	avgs=rst[0]
	dtss = rst[1]
	#avgs = rst[2]
	print len(avgs),len(dtss)
	for i in xrange(len(avgs)):
		avg=avgs[i]
		dt.draw([avg],c=dt.color(i),s =30)
		tdts = dtss[i]
		print len(tdts)
		if len(tdts)==0:
			continue 
		dt.draw(tdts, c = dt.color(i), s = 1)
	dt.show()
	if not nxt:
		return None
	#return None
	tree = cluster.cluster(dts,k, cluster.min_dsts)
	print "done tree"
	sets = cluster.tree2sets(tree)
	#sets = cluster.cut_tree(tree,k,cluster.min_cost)
	print "done cut"
	print len(sets)
	for i in xrange(len(sets)):
		tdts = sets[i]
		print len(tdts)
		if len(tdts)==0:
			continue 
		dt.draw(tdts, c = dt.color(i), s = 1)
	dt.show()
	return sets
"""

python
from dtm import demo
demo.init(5)
demo.show()
rst = demo.test(5)

rst=demo.test(mx=False),demo.test(mx=True)

"""