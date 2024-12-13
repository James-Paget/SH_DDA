#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:03:35 2022

@author: phsh
"""

import scipy.special as sp


order = -1
scale = 0.000001
for i in range(-10,10):
    kt = 1e1
    x = scale*i
    y = 0.5*(sp.jv(order-1,x*kt)-sp.jv(order+1,x*kt))*kt/x
    z = sp.jv(order,x)/x
    print(x,y,z)