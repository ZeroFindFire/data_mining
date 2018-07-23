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
	def __init__(self, rand = lambda :random.random(), rate = 0.3):
		self.rate = rate
		self.rand = rand
	def __call__(self, gen):
		l = len(gen)
		lp = int(l * self.rate)
		lp = max(lp, 1)
		for i in xrange(lp):
			index = random.randrange(0, l)
			gen[index] = self.rand()
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
				else:
					choice -= self.rate_cross 
					gen = list(random.choice(good_gens))
					if choice <= rate_mutate:
						gen = self.fc_mutate(gen)
					if gen not in new_gens:
						new_gens.append(gen)
			if best_gen is not None and best_gen[1] not in new_gens:
				new_gens.append(best_gen[1])
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

remain = """
class GAEncapeA(object):
	def __init__(self, fc_next_gens):
		self.__fc_next_gens = fc_next_gens 
		self.__fc_stop_judge = fc_stop_judge
	def run(self, fc_stop_judge):
		while not fc_stop_judge():
			self.__fc_next_gens()

class GAEncapeB(GAEncapeA):
	def __init__(self,  fc_select, fc_update):
		super(GAEncapeB, self).__init__(self.fc_next_gens)
		self.__fc_selelct = fc_select 
		self.__fc_update = fc_update
	def fc_next_gens(self):
		self.__fc_update()

"""

[0]==[0]

class ConstRate(object):
	def __init__(self, rate):
		self.rate = rate 
	def __call__(self):
		return self.rate 

class AlterRate(object):
	@staticmethod
	def default_cal(rate, inc, count):
		import numpy 
		return rate + inc * numpy.log(count**2 + 1)
	def __init__(self, rate, increment, fc_best_gen, 
		 fc_rate_cal = None, max_rate = 0.5):
		if fc_rate_cal is None:
			fc_rate_cal = AlterRate.default_cal
		self.rate = rate 
		self.increment = increment
		self.best_gen = None
		self.count = 0
		self.max_rate = max_rate
		self.fc_rate_cal = fc_rate_cal
		self.fc_best_gen = fc_best_gen
	def __call__(self):
		best_gen = self.fc_best_gen()
		if self.best_gen is None or best_gen != self.best_gen:
			self.best_gen = best_gen
			self.count = 0
		else:
			self.count += 1
		rate = self.fc_rate_cal(self.rate , self.increment , self.count)
		rate = min(rate, self.max_rate)
		return rate 

class GAEncapeC(object):
	def __init__(self, gens, num_gens, fc_select, fc_cross, fc_mutate, fc_rate_mutate):
		#super(GAEncapeC, self).__init__(self.fc_select, self.fc_update)
		self.__fc_select = fc_select
		self.__fc_cross = fc_cross
		self.__fc_mutate = fc_mutate
		self.__num_gens = num_gens
		self.__fc_rate_mutate = fc_rate_mutate
		self.__gens = gens
	def fc_update(self, gens):
		gens = self.__fc_select(gens)
		next_gens = []
		rate_mutate = self.__fc_rate_mutate()
		while len(next_gens) < self.__num_gens:
			rand = random.random()
			if rand < rate_mutate:
				new_gens = [self.__fc_mutate(gens)]
			else:
				new_gens = list(self.__fc_cross(gens))
			next_gens += new_gens 
		return next_gens 
	def run(self, fc_stop_judge):
		while not fc_stop_judge():
			self.__gens = self.fc_update(self.__gens)

class GAEncapeD(GAEncapeC):
	def __init__(self, gens, num_gens, fc_select, fc_cross, fc_mutate, fc_best_gen, fc_rate_mutate):
		super(GAEncapeD, self).__init__(gens, num_gens, fc_select, self.fc_cross, self.fc_mutate, fc_rate_mutate)
		self.__fc_best_gen = fc_best_gen
		self.__best_gen = fc_best_gen(gens)
		self.__fc_cross = fc_cross 
		self.__fc_mutate= fc_mutate
		self.__fc_rate_mutate = fc_rate_mutate
	def fc_rate_mutate(self):
		best_gen = self.__best_gen 
		return self.__fc_rate_mutate()
	def fc_update(self,gens):
		next_gens = super(GAEncapeD, self).fc_update(gens)
		if self.__best_gen not in next_gens:
			next_gens.append(self.__best_gen)
		self.__best_gen = self.__fc_best_gen(next_gens)
		return next_gens
	def best_gen(self):
		return self.__best_gen
	def __call__(self):
		return self.best_gen()
	def fc_cross(self, gens):
		#print "gens:",len(gens)
		gens = [random.choice(gens)[:] for i in [0,0]]
		return self.__fc_cross(gens)
	def fc_mutate(self, gens):
		#print "gens:",len(gens)
		gen = random.choice(gens)[:]
		return self.__fc_mutate(gen)
class GAEncapeValue(GAEncapeD):
	def __init__(self, gens, num_gens, fc_select, fc_cross, fc_mutate, fc_rate_mutate, fc_score):
		self.__fc_select = fc_select 
		self.__fc_score = fc_score 
		super(GAEncapeValue, self).__init__(gens, num_gens, self.fc_select, fc_cross, fc_mutate, self.fc_best_gen, fc_rate_mutate)
	def fc_select(self, gens):
		scores = self.__fc_score(gens)
		indexs = self.__fc_select(scores)
		return [gens[i] for i in indexs]
	def fc_best_gen(self, gens):
		scores = self.__fc_score(gens)
		import numpy 
		index = numpy.argmax(scores)
		return gens[index]

class ScoreSelector(object):
	@staticmethod
	def rand_select(scores, num):
		outs = []
		total_score = sum(scores)
		l = len(scores)
		if num < l:
			scores = scores[:]
		for k in xrange(num):
			lgen = len(scores)
			rand_score = random.random() * total_score
			for i in xrange(lgen):
				rand_score -= scores[i]
				if rand_score * total_score <= 0.0:
					outs.append(i)
					if num < l:
						total_score-=scores.pop(i)
					break
		return outs
	@staticmethod
	def rand_select_adapt(scores, num):
		outs = []
		total_score = sum(scores)
		lgen = len(scores)
		tmp = list(scores)
		tmp.sort(reverse = True)
		avg_score = tmp[lgen/2] 
		up_ids, up_scores, down_ids, down_scores = [],[],[],[]
		for i in range(lgen):
			if scores[i] >= avg_score:
				up_ids.append(i) 
				up_scores.append(scores[i])
			else:
				down_ids.append(i) 
				down_scores.append(scores[i])
		ups = ScoreSelector.rand_select(up_scores, num/2)
		downs = ScoreSelector.rand_select(down_scores, num/2)
		outs = [up_ids[i] for i in ups] + [down_ids[i] for i in downs]
		return outs

	@staticmethod
	def sort_select(scores, num):
		ids = [[scores[i],i] for i in xrange(len(scores))]
		ids.sort(key = lambda x:x[0], reverse = True)
		outs = [i for s,i in ids]
		return outs[:num]

	def __init__(self, fc_select, num):
		self.num = num 
		self.fc_select = fc_select 
	def __call__(self, scores):
		#print "scores:",scores
		indexs = self.fc_select(scores, self.num)
		return indexs

class CacheScoreCal(object):
	def __init__(self, fc_score, cache_size = 100):
		self.cache_size = cache_size 
		self.fc_score = fc_score 
		self.caches_gen = []
		self.caches_score = []
	def __call__(self, gens):
		indexs = []
		for gen in gens:
			if gen in self.caches_gen:
				index = self.caches_gen.index(gen)
			else:
				score = self.fc_score(gen)
				self.caches_gen.append(gen)
				self.caches_score.append(score)
				index = len(self.caches_gen) - 1
			indexs.append(index)
		scores = [self.caches_score[index] for index in indexs]
		if len(self.caches_score) > self.cache_size:
			self.caches_gen = self.caches_gen[-cache_size:]
			self.caches_score = self.caches_score[-cache_size:]
		return scores
class StopJudger(object):
	def __init__(self, loop = None, same_loop = None, fc_best_gen = None):
		if loop is None and same_loop is None:
			raise Exception("loop and same_loop both None is not allow")
		if same_loop is not None and fc_best_gen is None:
			raise Exception("same_loop should use fc_best_gen, so fc_best_gen not allow to be None")
		self.loop = loop 
		self.same_loop = same_loop
		self.count = 0
		self.same_count = 0
		self.fc_best_gen = fc_best_gen
		if fc_best_gen is not None:
			self.best_gen = None 
	def __call__(self):
		self.count += 1
		if self.loop is not None and self.count > self.loop:
			return True 
		if self.same_loop is None:
			return False 
		best_gen = self.fc_best_gen() 
		if self.best_gen == best_gen:
			self.same_count += 1
		else:
			self.best_gen = best_gen
			self.same_count = 0
		if self.same_count > self.same_loop:
			return True 
		return False

class delay_best_gen(object):
	def __init__(self, fc_best_gen = None):
		self.fc_best_gen = None 
	def bind(self,fc_best_gen):
		self.fc_best_gen = fc_best_gen 
	def __call__(self):
		return self.fc_best_gen()
def create_ga(gens, nums_gen, fc_select, fc_cross, fc_mutate, fc_rate_mutate, fc_score, fc_best_gen = None):
	ga = GAEncapeValue(gens, nums_gen, fc_select, fc_cross, fc_mutate, fc_rate_mutate, fc_score)
	if fc_best_gen is not None:
		fc_best_gen.bind(ga)
	return ga 
demo = """

python
from dtm.ga import *
import numpy as np 
import random
mx = np.array([
[0,1,0],
[0,2,0],
[0,3,0]
])
def fc_score(gen):
	global mx 
	rst = 0
	pos = np.array([0,0])
	mv = np.array([[0,1],[1,0],[0,-1],[-1,0]])
	for stp in gen:
		pos += mv[stp]
		if (pos <0).sum()>0:
			return 0
		elif (pos >= mx.shape).sum()>0:
			return 0
		rst += mx[pos[0],pos[1]]
	return rst 

fc_scores = CacheScoreCal(fc_score,100)
nums_gen  = 10
fc_select = ScoreSelector(ScoreSelector.rand_select, nums_gen)
fc_cross = BaseCross()
fc_mutate = BaseMutate(rand = lambda :random.randint(0,3))
fc_best_gen = delay_best_gen()
fc_rate_mutate = AlterRate(0.3,0.01, fc_best_gen)
gen = [0,0,0]
gens = [gen for i in xrange(10)]
ga = create_ga(gens,nums_gen, fc_select, fc_cross, fc_mutate, fc_rate_mutate, fc_scores, fc_best_gen)
ga.run(StopJudger(10)),ga.best_gen(),fc_score(ga.best_gen())





"""
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

