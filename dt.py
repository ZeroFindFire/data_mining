#coding=utf-8
import matplotlib.pyplot as plt
import random
import numpy as np
def plots(pts):
	x = [p[0] for p in pts]
	y = [p[1] for p in pts]
	return x,y

def draw(pts,c='red',s=10):
	x,y = plots(pts)
	plt.scatter(x,y,c=c,s=s)

def show():
	plt.show()

def show_pt(pts,c='red',s=10):
	draw(pts,c=c,s=s)
	show()
def rand(rd):
	return (random.random() - 0.5) * rd

def circles(x,y,r,num, flat = 0.0):
	outs = []
	cx,cy = x,y
	for i in xrange(num):
		tr = r + rand(flat)
		angle = random.random() * 2 * np.pi
		x,y = tr*np.cos(angle), tr*np.sin(angle)
		outs.append([cx+x,cy+y])
	return outs

def point(x,y, num, rx, ry):
	outs = []
	for i in xrange(num):
		angle = random.random() * 2 * np.pi
		tx = x + rand(rx)*np.cos(angle)
		ty = y + rand(ry)*np.sin(angle)
		outs.append([tx,ty])
	return outs


def toplt(func, *attrs):
	pts = func(*attrs)
	x,y = plots(pts)
	return x,y


cls = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
l = len(cls)
for i in xrange(l):
	cls[i] = np.array(cls[i],dtype = np.float)
for t in xrange(3):
	l = len(cls)
	for i in xrange(l):
		for j in xrange(i+1,l):
			c = (cls[i] + cls[j]) * 0.5
			cls.append(c)

def color(index):
	l = len(cls)
	index %= l 
	return cls[index]
	r = index & 0xFF
	index >>= 8
	g = index & 0xFF
	index >>= 8
	b = index & 0xFF

	pass
"""

python
from dtm import dt

"""