#coding=utf-8

# fc_done(avgs, sets, old_avgs, old_sets, count)
# 
class DefaultDone:
	def __init__(self,max_count = 100):
		self.count = max_count
		pass 
	def __call__(self,avgs, sets, old_avgs, old_sets, count):
		if count >= self.count:
			return True 

		for i in xrange(len(avgs)):
			if (avgs[i]!= old_avgs[i]).sum()>0:
				return False 
		return True

import numpy as np
def k_means(data, k, fc_done = DefaultDone()):
	data = np.asarray(data,dtype = np.float64)
	pts = np.array(data[:k])
	count = 0
	sets = [[] for pt in pts]
	while True:
		old_sets = sets 
		sets = [[] for pt in pts]
		for dt in data:
			tpts = np.abs(pts - dt )
			tpts = tpts.sum(axis=1)
			id = np.argmin(tpts)
			sets[id].append(dt)
		old_pts = np.array(pts)
		for i in xrange(k):
			if len(sets[i])==0:
				pts[i]=np.array([0,0],dtype = np.float64)
				continue 
			pts[i] = sum(sets[i]) * 1.0 / len(sets[i])
		if fc_done(pts, sets, old_pts, old_sets, count):
			break 
		count +=1
	return pts, sets
def default_dst(a,b):
	return np.abs(a-b).sum()

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
	# len(dsts) = l * (l-1) / 2
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
def default_cost(tree):
	if type(tree) != list:
		return 0.0
	dts = to_list(tree)
	l = len(dts)
	dst = 0.0
	for i in xrange(l):
		for j in xrange(i+1,l):
			dst += default_dst(dts[i],dts[j])
	return dst * 1.0 / (l * (l-1) * 0.5)
def min_cost(tree):
	if type(tree) != list:
		return 0.0
	dts = to_list(tree)
	l = len(dts)
	dst = 0.0
	for i in xrange(l):
		dst += min([default_dst(dts[i],dts[j]) for j in xrange(l) if i!=j])
	return dst * 1.0 / (l * (l-1) * 0.5)

def cut_tree(tree, k, fc_cost = default_cost):
	cuts = [tree]
	outs = []
	while len(cuts) < k:
		nd, cst = None, None
		for t in cuts:
			tcst = fc_cost(t)
			if cst is None or tcst > cst:
				nd, cst = t, tcst 
		if type(nd) != list:
			break
		if len(nd) != 2:
			print "not right?:",nd
			print len(cuts)
			break
		cuts.remove(nd)
		cuts.append(nd[0])
		cuts.append(nd[1])
	for c in cuts:
		outs.append(to_list(c))
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
