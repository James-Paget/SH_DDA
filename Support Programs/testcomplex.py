#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:11:12 2022

@author: phsh
"""

import numpy as np
from numpy.random import rand
# Randomly choose real and imaginary parts.
# Treat last axis as the real and imaginary parts.
A = rand(100, 2)
# Cast the array as a complex array
# Note that this will now be a 100x1 array
A_comp = A.view(dtype=np.complex128)
# To get the original array A back from the complex version
A = A.view(dtype=np.float64)

B = np.linspace(0,17,18)

C=B.reshape((9,2)).view(dtype=np.complex128).reshape((3,3))
C=B.view(dtype=np.complex128).reshape((3,3))

D = C.view(dtype=np.float64).reshape((18,1)).flatten()
