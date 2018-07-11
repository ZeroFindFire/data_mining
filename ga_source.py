#coding=utf-8

# GA:
# cal(groups, fc_score, max_time, *attrs)
# 
# step:
# 	score:
# 		scores = [fc_score(gem) for gem in groups]
# 	select:
# 		randomly select a few good gem in good scores
# 	cross:
# 		select 2 gem to do cross
# 	change:
# 		select gem to do change
# 	select:
# 		randomly select good gem in good scores in all gem
# 		
# 		

import random
class BaseCross(object):
	def __call__(self, gens):
		l = len(gens[0])
		i = random.randrange(0, l - 1)
		j = random.randrange(i, l)
		parts = [gen[i:j+1] for gen in gens]
		for index in xrange(2):
			gens[index][i : j+1] = parts[1 - index]
		return gens

class BaseMutate(object):
	def __init__(self, bound, rate = 0.3):
		self.rate = rate
		self.bound = bound
	def __call__(self, gen):
		l = len(gen)
		lp = int(l * self.rate)
		for i in xrange(lp):
			index = random.randrange(0, l)
			gen[index] = random.randint(self.bound)
		return gen

class OrderCross(object):
	def __init__(self, include_bound = False):
		self.include_bound = include_bound
	def __call__(self, gens):
		l = len(gens[0])
		if not self.include_bound:
			base = 1
			last = l-1
		else:
			base = 0
			last = l
		i = random.randrange(base, last - 1)
		j = random.randrange(i, last)
		parts = [gen[i:j+1] for gen in gens]
		pos = set(xrange(i, j+1))
		# print pos
		left = set(xrange(base,last)) - pos
		change_mark = True 
		while change_mark:
			change_mark = False
			for index in list(left):
				for mark in (0,1):
					if gens[mark][index] in parts[1-mark]:
						pos.add(index)
						left.remove(index)
						parts[mark].append(gens[mark][index])
						parts[1-mark].append(gens[1-mark][index])
						change_mark = True
						break
		# print pos
		for index in pos:
			gens[0][index], gens[1][index] = gens[1][index], gens[0][index]
		return gens

class OrderMutate(object):
	def __init__(self, rate = 0.3, include_bound = False):
		self.include_bound = include_bound
		self.rate = rate
	def __call__(self, gen):
		l = len(gen)
		if not self.include_bound:
			base = 1
			last = l-1
		else:
			base = 0
			last = l
		lp = int(l * self.rate)
		lp = max(lp, 1)
		lp = random.randint(1, lp)
		for i in xrange(lp):
			index0 = random.randrange(base, last - 1)
			index1 = random.randrange(index0 + 1, last)
			gen[index0], gen[index1] = gen[index1], gen[index0]
		return gen
import numpy as np
def rmatrix(size):
	matrix = np.random.randint(0,256, [size, size])
	return matrix
class Test(object):
	def score(self, gen):
		#if len(gen) != len(self.matrix):
		#	raise Exception("len(gen) != self.matrix: %d != %d" %(len(gen), len(self.matrix)))
		rst = 0
		for i in xrange(len(gen) - 1):
			rst += self.matrix[gen[i]][gen[i+1]]
		return -rst
	def scores(self,gens):
		outs = [self.score(gen) for gen in gens]
		return outs
	def __init__(self, size = 10, gens = 20, heritable = 0.3, matrix = None):
		# best: 0, 1, 3, 2
		
		self.matrix = [
		[0, 1, 6, 9],
		[0, 0, 4, 1],
		[3, 1, 0, 7],
		[1, 2, 5, 0],
		]
		self.inits = [
		[0, 1, 2, 3, 0],
		[0, 2, 1, 3, 0],
		[0, 2, 3, 1, 0],
		]
		if matrix is not None:
			self.matrix = matrix
			size = len(matrix)
		else:
			self.matrix = np.random.randint(0,256, [size, size])
		self.inits = [
		range(size) + [0]
		]
		self.cross = OrderCross()
		self.mutate = OrderMutate()
		self.ga = GA(self.inits, self.scores, self.cross, self.mutate, num_gens = 20, num_heritable = heritable)

class GA(object):
	@staticmethod
	def rand_select(scores, gens, num):
		outs = []
		total_score = sum(scores)
		if len(scores) != len(gens):
			raise Exception("len(scores) != len(gens): %d != %d" % (len(scores), len(gens)))
		lgen = len(scores)
		for k in xrange(num):
			rand_score = random.random() * total_score
			for i in xrange(lgen):
				rand_score -= scores[i]
				if rand_score * total_score < 0.0:
					outs.append(gens[i])
					break
		return outs
	@staticmethod
	def rand_select_adapt(scores, gens, num):
		outs = []
		total_score = sum(scores)
		if len(scores) != len(gens):
			raise Exception("len(scores) != len(gens): %d != %d" % (len(scores), len(gens)))
		lgen = len(scores)
		tmp = list(scores)
		tmp.sort()

		avg_score = tmp[lgen/2] #1.0 * total_score / (lgen+0.1)
		up_gens, up_scores, down_gens, down_scores = [],[],[],[]
		for i in range(lgen):
			if scores[i] >= avg_score:
				up_gens.append(gens[i]) 
				up_scores.append(scores[i])
			else:
				down_gens.append(gens[i]) 
				down_scores.append(scores[i])
		ups = GA.rand_select(up_scores, up_gens, num/2)
		downs = GA.rand_select(down_scores, down_gens, num/2)
		return ups+downs

	@staticmethod
	def sort_select(scores, gens, num):
		if len(scores) != len(gens):
			raise Exception("len(scores) != len(gens): %d != %d" % (len(scores), len(gens)))
		ids = [[scores[i],i] for i in xrange(len(scores))]
		ids.sort(key = lambda x:x[0], reverse = True)
		outs = [gens[i] for s,i in ids]
		return outs[:num]


	# fc_score(gens): return scores of gens
	# fc_select(scores, gens, num): select number num of gens according to their scores
	# fc_cross(gens[2]): return cross gens[2] from source gens[2]
	# fc_mutate(gen): return mutated gen from gen
	def __init__(self, gens, fc_score, fc_cross, fc_mutate, fc_select = None, 
		num_gens = -1, num_heritable = 0.3, rate_cross=0.3, rate_mutate=0.1, 
		disater_rate = 0.01, fc_disater = lambda count,mutate_rate,disater_rate:mutate_rate+disater_rate * np.log(count**2+1)):
		if fc_select is None:
			fc_select = self.rand_select_adapt
		if num_gens < 0:
			num_gens = len(gens)
		if type(num_heritable) == float:
			num_heritable = int(num_gens * num_heritable)
		self.rate_mutate = rate_mutate 
		self.rate_cross = rate_cross 
		self.num_heritable = num_heritable 
		self.num_gens = num_gens 
		self.fc_mutate = fc_mutate 
		self.fc_cross = fc_cross 
		self.fc_select = fc_select 
		self.fc_score = fc_score 
		self.gens = gens 
		self.best_gen = None
		self.scores = fc_score(self.gens)
		self.disater_rate = disater_rate
		self.disater_count = 0
		self.fc_disater = fc_disater



	def run(self, max_time = None, best_score = None):
		if max_time is None and best_score is None:
			return self.best_gen, 0
		mark_stop = False 
		count = 0
		best_gen = self.best_gen 
		gens = self.gens 
		while not mark_stop:
			new_gens = []

			good_gens = self.fc_select(self.scores, self.gens, self.num_heritable)
			self.tmp = good_gens
			while len(new_gens) < self.num_gens:
				rate_mutate = self.fc_disater(self.disater_count, self.rate_mutate, self.disater_rate)
				rate_mutate = min(1.0 - self.rate_cross, rate_mutate)
				choice = random.random() * (self.rate_cross + rate_mutate)
				if choice <= self.rate_cross:
					tuple_gen = [list(random.choice(good_gens)) for i in range(2)]
					tuple_gen = self.fc_cross(tuple_gen)
					for gen in tuple_gen:
						if gen not in new_gens:
							new_gens.append(gen)
					#new_gens += tuple_gen 
				else:
					choice -= self.rate_cross 
					gen = list(random.choice(good_gens))
					if choice <= rate_mutate:
						gen = self.fc_mutate(gen)
					if gen not in new_gens:
						new_gens.append(gen)
			if best_gen is not None and best_gen[1] not in new_gens:
				new_gens.append(best_gen[1])
			"""tmp_gens = new_gens
			new_gens = []
			for gen in tmp_gens:
				if gen not in new_gens:
					new_gens.append(gen)"""
			new_scores = self.fc_score(new_gens)
			self.gens = new_gens 
			self.scores = new_scores 
			max_score = max(new_scores)

			id_score = [i for i in xrange(len(new_scores)) if new_scores[i] == max_score][0]

			if best_gen is None or max_score > best_gen[0]:
				best_gen = [max_score, list(new_gens[id_score])]
				self.disater_count = 0
			else:
				self.disater_count += 1

			self.best_gen = best_gen 
			count += 1
			if best_score is not None and max_score >= best_score:
				mark_stop = True
			elif max_time is not None:
				if count >= max_time:
					mark_stop = True

		return self.best_gen, count

"""
D(s,S) = min( [ D(i, S) + matrix[i,s] for i in range(1,n)])
D(i,S) = min( [ D(k, S-i) + matrix[k,i] for k in range(1,n) if k!= i])
D(i,S) = min( [ D(k, S-i) + matrix[k,i] for k in S-i])
"""
def translate(cap, size):
	index = 0
	for i in xrange(1, size):
		index = index << 1
		if cap[i]:
			index += 1
	return index
def dst(i, cap, matrix, mem, size, max_cal):
	dst.time+=1
	dst.mem = mem
	index = translate(cap, size)
	if mem[i, index] >= 0:
		return mem[i, index]
	if max_cal[0] > 0:
		max_cal[0]-=1 
	elif max_cal[0] == 0:
		curr, total = (mem!=-1).sum(),mem.size
		raise Exception("Time out, already cal: %d/%d, %f"%(curr, total, 1.0*curr / total) )
	rst = None
	tmpid = 0
	for j in xrange(size):
		if not cap[j]:
			continue 
		cap[j] = 0
		tmp = dst(j, cap, matrix, mem, size, max_cal) + matrix[j, i]
		cap[j] = 1
		if rst is None or tmp < rst:
			rst = tmp 
			tmpid = j
	if rst is None:
		print i, index, cap
	mem[i, index]  = rst
	return rst
def ttranslate(cap,size):
	tcap = list(range(size))
	tcap[0] = 0
	for i in cap:
		tcap[i] = 1
	return translate(tcap, size)


def dst_inv(i, cap, matrix, mem, size, bk = 1):
	dst_inv.time +=1
	dst_inv.mem = mem
	index = translate(cap, size)
	if index in mem[i] >= 0:
		return mem[i][index]
	rst = None
	for j in xrange(size):
		if not cap[j]:
			continue 
		cap[j] = 0
		tmp = dst_inv(j, cap, matrix, mem, size, bk+1) + matrix[i, j]
		cap[j] = 1
		if rst is None or tmp < rst:
			rst = tmp 
	mem[i][index]  = rst
	return rst

#dst_inv_cut_count = 0
def dst_inv_cut(i, cap, matrix, mem, size, left = None, bk = 1):
	dst_inv_cut.time +=1
	dst_inv_cut.mem = mem
	lcap = [k for k in xrange(1,size) if cap[k] == 1]
	index = translate(cap, size)
	#if len(lcap)==0 and (index in mem[i] or mem[i,index] != matrix[i,0]):
	#	raise Exception("len(lcap)==0 and mem[i, index]")
	#if mem[i, index] >= 0:
	if index in mem[i]:
		#return mem[i, index]
		tmp = mem[i][index]
		if left is not None and left <= tmp:
			return None
		return tmp

	rst = None

	zcap = lcap+[0]
	min_cal = sum([matrix[k][zcap].min() for k in lcap])
	cost = min_cal + matrix[i].min()
	if left is not None and cost >= left:
		return None
	tmp_cap = list(lcap)
	for k in lcap:
		j = random.choice(tmp_cap)
		tmp_cap.remove(j)
		#j=k
		if left is not None:
			cost = matrix[i,j] + min_cal
			if cost >= left:
				continue
		cap[j] = 0
		cut_left = left-matrix[i,j] if left is not None else None
		tmp = dst_inv_cut(j, cap, matrix, mem, size, cut_left, bk+1)
		cap[j] = 1
		if tmp is not None:
			tmp += matrix[i, j]
			if rst is None or (tmp < rst):
				rst = tmp
		if rst is not None:
			if left is None or left > rst:
				left = rst
	if rst is not None and rst <= left:
		mem[i][index] = rst
	return rst
def dist_inv(i, cap, matrix, mem, size, dst = dst, left = None):
	cap[i] = 0
	if left is None:
		rst = dst(i, cap, matrix, mem, size)
	else:
		rst = dst(i, cap, matrix, mem, size, left = left)
	cap[i] = 1
	return rst

def dsp_inv(matrix, reset = True, dst = dst_inv, left = None):
	dst.time = 0
	dst.order=[]
	size = len(matrix)
	msize = 2 ** (size-1)
	if reset:
		#mem = np.zeros((size, msize), dtype = np.int) 
		#mem[:] = -1
		mem = [dict() for i in xrange(size)]
	else:
		mem = dsp_inv.mem
	for i in xrange(1, size):
		mem[i][0] = matrix[i, 0]
	dsp_inv.mem = mem 
	cap = np.ones(size, dtype = np.int)
	rst = dist_inv(0, cap, matrix, mem, size, dst, left)
	#return rst
	tmprst = rst 
	curr = 0
	outs = [0]
	cap = np.ones(size, dtype = np.int)
	for time in xrange(size-1):
		for i in xrange(1,size):
			if not cap[i]:
				continue
			cap[i]=0
			index = translate(cap,size)
			cap[i]=1
			if index not in mem[i]:
				continue
			val = mem[i][index]
			if val == -1:
				continue
				#print "error",i,index,cap
			#val = dist_inv(i, cap, matrix, mem, size, dst)
			if val + matrix[curr,i] == tmprst:
				outs.append(i)
				cap[i] = 0
				tmprst -= matrix[curr, i]
				curr = i
				break 
	outs.append(0)
	return rst,outs
"""
mem[i,0] = matrix[i,0]
D(i,empty):
D(0,S-0) = min(D(i, S-0-i)+mx[i,0]) = min(mx[0,i] + D(i->0, S-i-0))

D(i->0, S-i-0) = min(mx[i,j] + D(j->0, S-i-j-0))

D(i,S):从i到0，遍历S中节点
"""
def cut_dst(i, cap, matrix, mem, size, max_cal, left = None, bk = 1):
	index = translate(cap, size)
	blk = ("% "+str(bk*8)+"d") %(0,)
	print blk,"i ",i,"cap:",cap,"index:",index
	if mem[i, index] >= 0:
		print blk,"Mi ",i,"cap:",cap,"index:",index,"mem:",mem[i, index]
		return mem[i, index]
	if max_cal[0] > 0:
		max_cal[0]-=1 
	elif max_cal[0] == 0:
		curr, total = (mem!=-1).sum(),mem.size
		raise Exception("Time out, already cal: %d/%d, %f"%(curr, total, 1.0*curr / total) )
	rst = None

	lcap = [k for k in xrange(1,size) if cap[k] == 1]

	zcap = lcap + [0]
	icap = lcap + [i]
	min_cal = sum([matrix[k][lcap].min() for k in zcap])

	cost = min_cal + matrix[i].min()
	if left is not None and cost > left:
		mem[i, index]  = matrix.sum()
		return None

	for j in lcap:
		if left is not None:
			min_cal = sum([matrix[k][icap].min() for k in zcap])
			cost = matrix[j,i] + min_cal
			if cost > left:
				continue
		cap[j] = 0
		cut_left = left-matrix[i,j] if left is not None else None
		#cut_left = None
		tmp = cut_dst(j, cap, matrix, mem, size, max_cal, cut_left,bk+1)
		if tmp is not None:
			tmp += matrix[j, i]
		cap[j] = 1

		if tmp is not None:
			if rst is None or (tmp < rst):
				rst = tmp 
				left = rst
		#if rst is not None:
		#	left = rst
		#left = None
	trst = rst if rst is not None else matrix.sum()

	mem[i, index]  = trst
	print blk,"Ri ",i,"cap:",cap,"index:",index,"mem:",mem[i, index]
	#if mem[i, index] == matrix.sum():
	#	print mem[i, index]
	return rst
def dist(i, cap, matrix, mem, size, max_cal, dst = dst):
	cap[i] = 0
	rst = dst(i, cap, matrix, mem, size, max_cal)
	cap[i] = 1
	return rst
def dsp(matrix, max_cal = -1, reset = True, dst = dst):
	dst.time = 0
	size = len(matrix)
	msize = 2 ** (size-1)
	max_cal = [max_cal] 
	if reset:
		mem = np.zeros((size, msize), dtype = np.int) 
		mem[:] = -1
	else:
		mem = dsp.mem
	for i in xrange(1, size):
		mem[i][0] = matrix[0, i]
	dsp.mem = mem 
	cap = np.ones(size, dtype = np.int)
	rst = dist(0, cap, matrix, mem, size, max_cal, dst)
	tmprst = rst 
	curr = 0
	outs = [0]
	cap = np.ones(size, dtype = np.int)
	for time in xrange(size-1):
		for i in xrange(1,size):
			if not cap[i]:
				continue
			val = dist(i, cap, matrix, mem, size, [-1], dst)
			if val + matrix[i,curr] == tmprst:
				outs.append(i)
				cap[i] = 0
				tmprst -= matrix[i,curr]
				curr = i
				break 
	outs.append(0)
	outs.reverse()
	return rst, outs

def cut_node(i, matrix, cap, left = None, blk = 1):
	cut_node.time+=1
	if len(cap) == 0:
		return matrix[i,0], [i,0]
	rst = None
	out_list = [-1]
	tmp_cap = list(cap)
	zcap = cap+[0]
	min_cal = sum([matrix[k][zcap].min() for k in cap])

	cost = min_cal + matrix[i].min()
	if left is not None and cost >= left:
		return rst, out_list
	for t in list(cap):
		j = random.choice(tmp_cap)
		tmp_cap.remove(j)
		#j=t
		if left is not None:
			cost = matrix[i,j] + min_cal
			if cost >= left:
				continue
		next_cap = list(cap)
		next_cap.remove(j) 
		cut_left = left-matrix[i,j] if left is not None else None
		tmp_rst, tmp_out_list = cut_node(j,matrix,next_cap,cut_left, blk+1)
		if tmp_rst is not None:
			tmp_rst+= matrix[i,j]
			if rst is None or (tmp_rst < rst):
				rst = tmp_rst
				out_list = [i] + tmp_out_list
		if rst is not None:
			if left is None or left > rst:
				left = rst
		#cap.append(j)
	return rst , out_list
def cut_tree(matrix, left = None):
	cut_node.order=[]
	cut_node.time = 0
	l = len(matrix)
	cap = list(range(1,l))
	min_mx = [x.min() for x in matrix]
	return cut_node(0,matrix, cap, left=left)

	pass


from testz import test
def tst(size = 0, mx = None):
	if mx is None:
		mx = ga.rmatrix(size)
		tst.mx = mx 
	rst = test.time(ga.dsp_inv,mx,True,ga.dst_inv_cut)
	print "dsp_inv_cut result:",rst
	rst = test.time(ga.cut_tree,mx)
	print "cut_tree    result:",rst
	rst = test.time(ga.dsp,mx)
	print "dsp         result:",rst
	rst = test.time(ga.dsp_inv,mx)
	print "dsp_inv     result:",rst
	return mx


def tst(size = 0, mx = None,cnt = 3):
	if mx is None:
		mx = rmatrix(size)
		tst.mx = mx 
	rst = test.time(dsp_inv,mx,True,dst_inv_cut)
	print "dsp_inv_cut result:",rst
	cnt -=1
	if cnt <=0:
		return mx 
	rst = test.time(cut_tree,mx)
	print "cut_tree    result:",rst
	cnt -=1
	if cnt <=0:
		return mx 
	rst = test.time(dsp,mx)
	print "dsp         result:",rst
	cnt -=1
	if cnt <=0:
		return mx 
	rst = test.time(dsp_inv,mx)
	print "dsp_inv     result:",rst
	return mx

def score( gen,matrix):
	#if len(gen) != len(self.matrix):
	#	raise Exception("len(gen) != self.matrix: %d != %d" %(len(gen), len(self.matrix)))
	rst = 0
	for i in xrange(len(gen) - 1):
		rst += matrix[gen[i]][gen[i+1]]
	return rst


class AntEncape(object):
	def __init__(self, ants_num, fc_single_ant, fc_update, fc_best_score = None):
		self.__ants_num = ants_num
		self.__fc_single_ant = fc_single_ant
		self.__fc_best_score = fc_best_score
		self.__fc_update = fc_update
		pass 
	def run(self, max_time = None, best_score = None):
		if max_time is None and best_score is None:
			return None
		mark_stop = False 
		count = 0
		while not mark_stop:
			rst = []
			for i in xrange(self.__ants_num):
				result = self.__fc_single_ant()
				rst.append(result)
			self.__fc_update(rst)
			if max_time is not None:
				count += 1
				if count >= max_time:
					mark_stop = True
			if best_score is not None:
				last_score = self.__fc_best_score()
				if last_score >= best_score:
					mark_stop = True
class TravelAnt(AntEncape):
	def run(self, max_time = None, best_score = None):
		super(type(self),self).run(max_time, best_score)
		return self.best_route
	def fc_update(self,scores):
		scores.append(self.best_route)
		mx = np.zeros([self.size,self.size])
		dv = 1.0 / (self.size + 1)
		max_score = max([score for score,route in scores]) * 1.1
		for obj in scores:
			score, route = obj 
			#score *= dv 
			#score = max_score - score
			score = 15.0 / score 
			for i in xrange(self.size):
				i_pos = route[i]
				mx[i, i_pos] += score 
		#mx *= 1.0 / len(scores)
		#for x in mx:
		#	x[:] *= 1.0 / x.sum()
		#mx = np.maximum(mx,0.1 * dv)
		#mx = np.minimum(mx,1.0)
		self.routes = self.routes * (1.0 - self.forgetten) + mx 
		self.routes = np.maximum(self.routes,0.1 * dv)
		self.routes = np.minimum(self.routes,1.0)
	def rand_select(self, index, pos):
		route = self.routes[index]
		total = sum([ route[i] for i in xrange(self.size) if i in pos[0] ])
		rand = np.random.rand() * total 
		new_pos = 0
		for new_pos in xrange(self.size):
			if new_pos not in pos[0]:
				continue 
			rand -= route[new_pos]
			if rand * total < 0.0:
				break
		return new_pos
	def fc_single_ant(self):
		pos = [ list(range(1, self.size)),[0]]
		score = 0
		for i in xrange(1, self.size):
			new_pos = self.rand_select(i, pos)
			score += self.matrix[pos[1][-1],new_pos]
			pos[0].remove(new_pos)
			pos[1].append(new_pos)
		score += self.matrix[pos[1][-1],0]
		pos[1].append(0)
		if self.best_route is None or self.best_route[0] > score:
			self.best_route = [score, pos[1]]
		return score, pos[1]
	def __init__(self, matrix, ants_num, forgetten = 0.5):
		super(type(self),self).__init__(ants_num, self.fc_single_ant, self.fc_update)
		self.matrix = matrix
		self.size = len(matrix)
		self.routes = np.random.random([self.size,self.size]) + 0.01
		self.routes[:,0]=0.0
		self.routes[0]=0.0
		self.best_route = None
		self.forgetten = forgetten
def test_ant(size,ants_num = 20, forgetten = 0.05, mx = None):
	if mx is not None:
		size = len(mx)
	else:
		mx = np.random.randint(0,256, [size, size])
	ant = TravelAnt(mx,ants_num,forgetten)
	return ant, mx


"""
	策略，实时，可操作，
	策略性：装备，属性互相克制
	操作互相克制
	白天黑夜
	天气：晴，阴，雨，刮风，气温
	野外地形
	村落
	人类，丧尸
	植物，动物
	武器，装备
	工具，建筑
	箱子，背包，挎包
	承载器具
	模拟真实世界
	背景：被恶魔侵占的大陆，各种种族生物，村落，城镇，
"""



def test(a,b):
	a.act(b)
	b.act(a)
	return a,b 

class Object(object):
	def __init__(self):
		self.life = np.array([100,100])
		self.power = np.array([60,60])
	









































def upobj(object):
	return super(type(object),object)

class Object(object):
	def init(self,*args):
		self.super().__init__(*args)
	def super(self):
		return super(type(self),self)
	def __init__(self, default = None):
		self.__default = default 
		print "type:",type(self)
	def __setattr__(self, name, value):
		return object.__setattr__(self, name, value)
	def __getattr__(self, name):
		if name in ['__dict__']:
			return object.__getattr__(self, name)
		elif name in self.__dict__:
			return self.__dict__[name]
		else:
			return self.__default
	def descript(self, n_words):
		return ""

def descript(self,n_words):
	pass 


"""
	分层:
	0: 描述
	descript(state, last_state)
	map:
	
"""

class Place(Object):
	def __init__(self, name):
		self.super().__init__(0)
		self.connects = set()
		self.objects = set()
		self.childs = set()
		self.description = name
	def descript(self, n_words):
		outs = self.description
		if self.childs is not None:
			outs += '\n'+'\n'.join([child.descript(n_words) for child in self.childs])
		if self.objects is not None:
			outs += '\n'+'\n'.join([child.descript(n_words) for child in self.objects])
		outs += "\n"
		return outs
	def connect(self,place):
		self.connects.add(place)
		place.connects.add(self)
	def put(self,obj):
		self.objects.add(obj)

def PlaceConnect(filename, maps = {}, split=" "):
	cts = rw.file_get_contents(filename)
	place_map = maps
	arr = cts.split("\n")
	cts_arr = [ct.strip() for ct in arr]
	for cts in cts_arr:
		names = [c.strip() for c in cts.split(split) if c.strip()!=""]
		if len(names) != 2:
			continue 
		for n in names:
			if n not in place_map:
				place_map[n] = Place(n)
		n0, n1 = names 
		place_map[n0].connect(place_map[n1])
	return place_map 

def ObjectPut(filename, maps = {}, split=" "):
	cts = rw.file_get_contents(filename)
	place_map = maps
	arr = cts.split("\n")
	cts_arr = [ct.strip() for ct in arr]
	for cts in cts_arr:
		names = [c.strip() for c in cts.split(split) if c.strip()!=""]
		if len(names) <2:
			continue 
		plc = names[0]
		names = names[1:]
		if plc not in place_map:
			place_map[plc] = Place(plc)
		for n in names:

			place_map[plc].put()
	pass






"""
place object list
placea placeb
...

"""

class Env(Object):
	def __init__(self):
		pass
class Life(Object):
	def __init__(self, place):
		self.super().__init__()
		self.pos = place
		place.object.add(self)

def run():
	pass
