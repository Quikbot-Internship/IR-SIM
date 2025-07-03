from __future__ import division
import numpy as np
from numpy import array
from numpy import sqrt, dot

def norm_sq(x):
        return dot(x, x)

def normalized(x):
    
    l = norm_sq(x)
    if l == 0:
        return np.zeros_like(x)  # return zero vector instead of crashing
    return x / sqrt(l)

def dist_sq(a, b):
    return norm_sq(b - a)


def perp(a):
    return array((a[1], -a[0]))

def norm(x):
    return sqrt(norm_sq(x))

    
def compute_pref_velocity(pos, goal, max_speed):
    direction = goal[:2].flatten() - pos
    
    #print("Direction vector:", direction)
    #print("Distance to goal:", np.linalg.norm(direction))

    dist = np.linalg.norm(direction)
    if dist < 0.01:
        return np.zeros(2)
    return (direction / dist) * min(max_speed, dist)