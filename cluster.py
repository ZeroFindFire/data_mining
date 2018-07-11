#coding=utf-8

# fc_done(avgs, sets, old_avgs, old_sets, count)
# 
class DefaultDone:
	def __init__(self,max_count = 100):
		self.count = max_count
		pass 
	def __call__(self,avgs, sets, old_avgs, old_sets, count):
		if count >= self.count:
			print "out of count"
			return True 

		for i in xrange(len(avgs)):
			if (avgs[i]!= old_avgs[i]).sum()>0:
				return False 
		print "no change"
		return True

import numpy as np
def sqr_dst(pts, pt):
	tpts = (pts - pt )**2
	tpts = tpts.sum(axis=1)
	return tpts
def sqr_cost(data, avgs):
	r = 0.0
	for dt in data:
		t = sqr_dst(avgs, dt).min()
		r+= t
	return r 
class CutDone:
	def __init__(self,data,n = 0.01, max_count = 1000):
		self.count = max_count
		self.data = np.array(data ,dtype = np.float64)
		self.n = n
		pass 
	def __call__(self,avgs, sets, old_avgs, old_sets, count):
		#print "shape:",self.data.shape,old_avgs.shape
		old_cost = sqr_cost(self.data, old_avgs)
		v_cost = sqr_cost(self.data, avgs)
		if old_cost == v_cost:
			print "cost:",v_cost 
			return True 
		if np.abs(((old_cost - v_cost) / old_cost)) < self.n:
			print "old_cost:",old_cost,"cost:",v_cost 
			return True 
		if count >= self.count:
			print "out of count"
			return True 
		return False
		for i in xrange(len(avgs)):
			if (avgs[i]!= old_avgs[i]).sum()>0:
				return False 
		print "no change"
		return True
def default_dst(pts, pt):
	tpts = np.abs(pts - pt )
	tpts = tpts.sum(axis=1)
	return tpts


def kernel_dst(pts, pt):
	ta = (pts **2).sum(axis=1) ** 2 
	tc = (pt ** 2).sum() ** 2
	tb = (pts * pt ).sum(axis=1) ** 2
	rst = ta - 2 * tb + tc 
	return rst

def k_means(data, k, fc_done = DefaultDone(), fc_dst = default_dst, mx = True):
	import random
	data = np.asarray(data,dtype = np.float64)
	len_data = len(data)
	if mx:
		pts = [data[random.randint(0,len_data-1)]]
		for i in xrange(k-1):
			tdst = np.array([0.])
			#print "tdst:",tdst.shape
			for pt in pts:
				ttdst = fc_dst(data, pt)
				#print "ttdst:",ttdst.shape
				tdst = tdst + ttdst 
			index = np.argmax(tdst)
			#print "tdst:",tdst
			pts.append(data[index])
		pts = np.array(pts,dtype = np.float64)
	else:
		pts = []
		ids = set()
		for i in xrange(k):
			index = random.randint(0,len_data-1)
			while index in ids:
				index = (index + 1) % k 
			ids.add(index)
			pts.append(data[index])
		pts = np.array(pts,dtype = np.float64)
		#pts = np.array(data[:k])
	#print "PTS:",pts
	count = 0
	sets = [[] for pt in pts]
	while True:
		old_sets = sets 
		sets = [[] for pt in pts]
		for dt in data:
			#tpts = np.abs(pts - dt )
			#tpts = tpts.sum(axis=1)
			tpts = fc_dst(pts,dt)
			id = np.argmin(tpts)
			sets[id].append(dt)
		old_pts = np.array(pts)
		for i in xrange(k):
			if len(sets[i])==0:
				print "error len(setss[i])==0:",i
				pts[i]=np.array([0,0],dtype = np.float64)
				continue 
			pts[i] = sum(sets[i]) * 1.0 / len(sets[i])
		count +=1
		if fc_done(pts, sets, old_pts, old_sets, count):
			break 
	print "COUNT:",count
	return pts, sets, old_pts

def to_list(tree):
	outs = []
	for nd in tree:
		if type(nd) == list:
			outs += to_list(nd)
		else:
			outs.append(nd)
	return outs
def avg_dsts(a,b):
	la, lb = to_list(a), to_list(b)
	dst = 0.0
	for a in la:
		for b in lb:
			dst += np.abs(a-b).sum()
	dst *= 1.0/(len(la) * len(lb))
	return dst

def min_dsts(a,b):
	la, lb = to_list(a), to_list(b)
	dst = None
	for a in la:
		for b in lb:
			tdst = np.abs(a-b).sum()
			if dst is None or dst > tdst:
				dst = tdst
	return dst
def max_dsts(a,b):
	la, lb = to_list(a), to_list(b)
	dst = None
	for a in la:
		for b in lb:
			tdst = np.abs(a-b).sum()
			if dst is None or dst < tdst:
				dst = tdst
	return dst
def cluster(data, nodes = 2, fc_dst=min_dsts):
	data = np.asarray(data,dtype = np.float64)
	sets = [[dt] for dt in data]
	l = len(sets)
	ld = l * (l - 1) / 2
	dsts = [[None for j in xrange(i+1,l)] for i in xrange(l)]
	while len(sets)>nodes:
		l = len(sets)
		id = None
		dst = None
		for i in xrange(l):
			for j in xrange(i+1, l):
				if dsts[i][j-i-1] is not None:
					tdst = dsts[i][j-i-1]
				else:
					tdst = fc_dst(sets[i],sets[j])
					dsts[i][j-i-1] = tdst
				if dst is None or tdst < dst:
					id = i,j 
					dst = tdst 
		i,j = id
		sets[i] = [sets[i],sets[j]]
		dsts[i] = [None for k in xrange(i+1,l)]
		for k in xrange(j):
			dsts[k].pop(j-k-1)
		dsts.pop(j)
		sets.pop(j)
	return sets

def clusters(data, nodes = 2, fc_dst=min_dsts):
	rst = cluster(data, nodes, fc_dst)
	return tree2sets(rst)


def cluster_tmp(data, nodes = 2, fc_cost=None):
	data = np.asarray(data,dtype = np.float64)
	sets = [[dt] for dt in data]
	l = len(sets)
	ld = l * (l - 1) / 2
	dsts = [[None for j in xrange(i+1,l)] for i in xrange(l)]
	while len(sets)>nodes:
		l = len(sets)
		id = None
		dst = None
		for i in xrange(l):
			for j in xrange(i+1, l):
				if dsts[i][j-i-1] is not None:
					tdst = dsts[i][j-i-1]
				else:
					tdst = fc_dst(sets[i],sets[j])
					dsts[i][j-i-1] = tdst
				if dst is None or tdst < dst:
					id = i,j 
					dst = tdst 
		i,j = id
		sets[i] = [sets[i],sets[j]]
		dsts[i] = [None for k in xrange(i+1,l)]
		for k in xrange(j):
			dsts[k].pop(j-k-1)
		dsts.pop(j)
		sets.pop(j)
	return sets

def tree2sets(tree):
	outs = []
	for t in tree:
		dts = to_list(t)
		outs.append(dts)
	return outs 
	

def judge(sets):
	pass



def k_means_bak(data, k, fc_dst, fc_avg, fc_done = DefaultDone()):
	pts = data[:k]
	count = 0
	sets = [[] for pt in pts]
	while True:
		old_sets = sets 
		sets = [[] for pt in pts]
		for dt in data:
			id =0
			dst = None
			for i in xrange(k):
				tdst = fc_dst(pts[i], dt)
				if dst is None or tdst < dst:
					dst = tdst
					id = i
			sets[id].append(dt)
		old_pts = pts[:]
		for i in xrange(k):
			pts[i] = fc_avg(sets[i])
		if fc_done(pts, sets, old_pts, old_sets, count):
			break 
		count +=1
	return pts, sets
