# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:22:08 2021

@author: zll, wzq

In this program, all matrixes and vectors are *numpy.mat* type, vectors are in shape(n, 1) for default.
"""

import numpy as np
import torch


def g_and_h(x, func):
    x = torch.tensor(x.astype(float), requires_grad=True).to(torch.float32)

    y = func(x, True)
    grad = torch.autograd.grad(y, x, retain_graph=True, create_graph=True)

    H = torch.tensor([])
    for anygrad in grad[0]:
        try:
            H = torch.cat((H, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
        except RuntimeError:
            H = torch.tensor(np.zeros(shape=(x.size()[0],x.size()[0])))
    return np.mat(grad[0].detach().numpy()).reshape((x.shape[0],1)), np.mat(H.view(x.size()[0], -1))


def dogleg(func, x, delta_max=1, delta=0.01, eta=0, k_max=100):
    """
    degleg optimizer

    Args:
        func (function): target function.
        gradient (function): gradient function of func.
        hessian (function): hessian matrix function of func.
        x (list): starting point.
        delta_max (float, optional): upper bound of delta. Defaults to 0.1.
        delta (float, optional): delta_0. Defaults to 0.01.
        eta (float, optional): lower bound of actual reduction rho. Defaults to 0.1.
        k_max (int, optional): # of iterations. Defaults to 1000.

    Returns:
        list type, [list of x, list of f(x)].
        returns all accepted iterations, from starting point to x*.

    """

    trace = [[np.mat(np.copy(x))], [func(x)]]  # to return
    for k in range(k_max):
        f = func(x)  # function value at this point

        g, B = g_and_h(x, func)# gradient vector of f at this point   # hessian matrix of f at this point.
        # x = np.mat(x.detach().numpy())
        def m(p):  # quadratic sub problem
            return float(f + g.T * p + 0.5 * p.T * B * p)

        try:
            p_B = -B.I * g
        except:  # if B is sigular, compute pseudo-inverse.
            p_B = -np.linalg.pinv(B) * g
        if np.linalg.norm(p_B) <= delta:
            p = p_B
        else:
            p_U = - float((g.T * g) / (g.T * B * g)) * g
            if np.linalg.norm(p_U) >= delta:
                p = p_U
            else:
                # solve quadratic function of (tau - 1)
                a = float((p_B - p_U).T * (p_B - p_U))
                b = float(2 * p_U.T * (p_B - p_U))
                c = float(p_U.T * p_U) - delta ** 2
                if b * b - 4 * a * c < 0:
                    print('cannot solve tau:', a, b, c, b * b - 4 * a * c)
                tau_1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)  # tau - 1
                if tau_1 < 0 or tau_1 > 1:
                    tau_1 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
                if tau_1 < 0 or tau_1 > 1:
                    print("can't find tau")
                p = p_U + tau_1 * (p_B - p_U)
        try:
            rho = (func(x) - func(x + p)) / (m(x - x) - m(p))  # actual reduction ratio
        except ZeroDivisionError:
            rho = eta + 1
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) >= 0.99 * delta:
            delta = min(2 * delta, delta_max)
        if rho > eta:
            x = p+x
        trace[0].append(np.mat(np.copy(x)))
        trace[1].append(func(x))
    return trace
