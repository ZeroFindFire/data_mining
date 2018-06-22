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

def k_means(data, k, fc_done = DefaultDone()):
	import numpy 
	data = numpy.asarray(data,dtype = numpy.float64)
	pts = numpy.array(data[:k])
	count = 0
	sets = [[] for pt in pts]
	while True:
		old_sets = sets 
		sets = [[] for pt in pts]
		for dt in data:
			tpts = numpy.abs(pts - dt )
			tpts = tpts.sum(axis=1)
			id = numpy.argmin(tpts)
			sets[id].append(dt)
		old_pts = numpy.array(pts)
		for i in xrange(k):
			if len(sets[i])==0:
				pts[i]=numpy.array([0,0],dtype = numpy.float64)
				continue 
			pts[i] = sum(sets[i]) * 1.0 / len(sets[i])
		if fc_done(pts, sets, old_pts, old_sets, count):
			break 
		count +=1
	return pts, sets

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
