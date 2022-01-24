# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:25:17 2021

@author: zll, wzq
"""


import numpy as np
from dogleg import g_and_h

def double_dogleg(func, x, delta_max=1, delta=0.01, eta=0, k_max=100):
    """
    double degleg optimizer

    Args:
        func (function): target function.
        gradient (function): gradient function of func.
        hessian (function): hessian matrix function of func.
        x (list): starting point.
        delta_max (float, optional): upper bound of delta. Defaults to 0.1.
        delta (float, optional): original trust rigion radius delta_0. Defaults to 0.01.
        eta (float, optional): lower bound of actual reduction rho. Defaults to 0.
        k_max (int, optional): # of iterations. Defaults to 100.

    Returns:
        list type, [list of x, list of f(x)].
        returns all accepted iterations, from starting point to x*.

    """
    trace = [[np.mat(np.copy(x))], [func(x)]]  # to return
    for k in range(k_max):
        f = func(x)      # function value at this point
        g, B = g_and_h(x, func) # gradient vector of f at this point.# hessian matrix of f at this point.
        def m(p):  # quadratic sub problem
            return float(f + g.T * p + 0.5 * p.T * B * p)
        try:  # inverse B
            BI = B.I
        except:  # if B is sigular, compute pseudo-inverse.
            BI = np.linalg.pinv(B)
        p_B = -BI * g
        p_C = - float((g.T * g) / (g.T * B * g)) * g
        if np.linalg.norm(p_B) <= delta:
            p = p_B
        elif np.linalg.norm(p_C) >= delta:
            p = delta * p_C / np.linalg.norm(p_C)
        else:
            gamma = float(np.linalg.norm(g) ** 4 / ((g.T * B * g) * (g.T * BI * g)))
            mu = 0.8 * gamma + 0.2
            p_N = mu * p_B
            # solve quadratic function of lambda
            a = float((p_N - p_C).T * (p_N - p_C))
            b = float(2 * p_C.T * (p_N - p_C))
            c = float(p_C.T * p_C) - delta ** 2
            if b * b - 4 * a * c < 0:
                print('cannot solve lambda:', a, b, c, b * b - 4 * a * c)
            lmd = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)  # lambda
            if lmd < 0 or lmd > 1:
                lmd = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            if lmd < 0 or lmd > 1:
                print("can't find lambda")
            p = p_C + lmd * (p_N - p_C)
        try:
            rho = (func(x) - func(x + p)) / (m(x - x) - m(p))  # actual reduction ratio
        except ZeroDivisionError:
            rho = eta + 1
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) >= 0.99 * delta:
            delta = min(2 * delta, delta_max)
        if rho > eta:
            x += p
        trace[0].append(np.mat(np.copy(x)))
        trace[1].append(func(x))
    return trace
