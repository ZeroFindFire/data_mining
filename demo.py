#coding=utf-8

import dt 
import cluster
import numpy as np 


dts = dt.circle(0,0,30,10,10) + dt.circle(20,20,30,10,10) + dt.circle(-20,60,30,10,10)
#dts = dt.circle(0,0,30,5,5) + dt.ring(0,0,20,200,5)
def test(count = 300, k = 3):
	global dts 
	dt.show_pt(dts,c=dt.color(0), s= 1)
	fc_done=cluster.DefaultDone(count)

	rst=cluster.k_means(dts,k,fc_done)

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
	tree = cluster.cluster(dts,k, cluster.max_dsts)
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
rst=demo.test()

"""