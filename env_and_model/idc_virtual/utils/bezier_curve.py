#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/3/26
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: bezier_curve.py.py
# =====================================
import numpy as np


class CubicBezierCurve:
    def __init__(self, p1, p2, p3, p4):
        self.p1x, self.p1y = p1[0], p1[1]
        self.p2x, self.p2y = p2[0], p2[1]
        self.p3x, self.p3y = p3[0], p3[1]
        self.p4x, self.p4y = p4[0], p4[1]

    def x(self, t):
        return self.p1x*np.power(1-t, 3) + 3*self.p2x*t*np.power(1-t, 2) + 3*self.p3x*np.power(t,2)*(1-t) + self.p4x*np.power(t,3)

    def y(self, t):
        return self.p1y*np.power(1-t, 3) + 3*self.p2y*t*np.power(1-t, 2) + 3*self.p3y*np.power(t,2)*(1-t) + self.p4y*np.power(t,3)

    def deriv_x(self, t):
        return 3*np.power(1-t,2)*(self.p2x-self.p1x) + 6*(1-t)*t*(self.p3x-self.p2x) + 3*np.power(t,2)*(self.p4x-self.p3x)

    def deriv_y(self, t):
        return 3*np.power(1-t,2)*(self.p2y-self.p1y) + 6*(1-t)*t*(self.p3y-self.p2y) + 3*np.power(t,2)*(self.p4y-self.p3y)

    def second_deriv_x(self, t):
        return 6*(1-t)*(self.p3x-2*self.p2x+self.p1x) + 6*t*(self.p4x-2*self.p3x+self.p2x)

    def second_deriv_y(self, t):
        return 6*(1-t)*(self.p3y-2*self.p2y+self.p1y) + 6*t*(self.p4y-2*self.p3y+self.p2y)

    def third_deriv_x(self, t):
        return -6*(self.p3x-2*self.p2x+self.p1x) + 6*(self.p4x-2*self.p3x+self.p2x)

    def third_deriv_y(self, t):
        return -6*(self.p3y-2*self.p2y+self.p1y) + 6*(self.p4y-2*self.p3y+self.p2y)

    def phi(self, t):
        return np.arctan2(self.deriv_y(t), self.deriv_x(t))

    def kappa(self, t):
        dx = self.deriv_x(t)
        dy = self.deriv_y(t)
        d2x = self.second_deriv_x(t)
        d2y = self.second_deriv_y(t)
        a = dx * d2y - dy * d2x
        b = np.power(dx * dx + dy * dy, 1.5)
        return a/b

    def dkappa(self, t):
        dx = self.deriv_x(t)
        dy = self.deriv_y(t)
        d2x = self.second_deriv_x(t)
        d2y = self.second_deriv_y(t)
        d3x = self.third_deriv_x(t)
        d3y = self.third_deriv_y(t)
        a = dx * d2y - dy * d2x
        b = dx * d3y - dy * d3x
        c = dx * d2x + dy * d2y
        d = dx * dx + dy * dy
        return (b * d - 3.0 * a * c) / np.power(d, 3.0)

    def _distance(self, x1, y1, x2, y2):
        return np.sqrt(np.power(x1-x2, 2) + np.power(y1-y2, 2))

    def length(self, t):
        l = 0.
        for i in range(999):
            if i/1000 > t:
                return l
            xi, yi = self.x(i/1000), self.y(i/1000)
            xip, yip = self.x((i+1)/1000), self.y((i+1)/1000)
            l += self._distance(xi, yi, xip, yip)
        return l
