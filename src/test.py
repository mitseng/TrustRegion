# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:23:59 2021

@author: zll, wzq

Achley's function, f*(0, 0) = 0, -5 <= x_i <= 5
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
from math import e, pi, sqrt
import torch
from dogleg import dogleg
from double_dogleg import double_dogleg

def f1(x, grad=False):
    if grad == False:
        return -20 * e ** float(-0.2 * np.sqrt(0.5 * x.T * x)) - e ** (0.5 * (np.sum(np.cos(2 * pi * x)))) + 20 + e
    else:
        return -20. * torch.exp(-0.2 * torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - torch.exp(
            0.5 * (torch.cos(2 * pi * x[0]) + torch.cos(2 * pi * x[1]))) + 20. + e


def f2(x, grad=False):
    if grad == False:
        return float(x.T * x)
    else:
        return x.t() @ x


def f3(x, grad=False):
    if grad == False:
        return np.sum(
            np.array([100 * (x[i + 1, 0] - x[i, 0] ** 2) ** 2 + (x[i, 0]) ** 2 for i in range(x.shape[0] - 1)]))
    else:
        A = [100 * (x[i + 1, 0] - x[i, 0] ** 2) ** 2 + (x[i, 0]) ** 2 for i in range(x.size()[0] - 1)]
        a = A[0]
        for i in A[1:]:
            a = a + i
        print(a)
        return a


def f4(x, grad=False):
    return (1.5 - x[0, 0] + x[0, 0] * x[1, 0]) ** 2 + (2.25 - x[0, 0] + x[0, 0] * x[1, 0] ** 2) ** 2 + (
            2.625 - x[0, 0] + x[0, 0] * x[1, 0] ** 3) ** 2


def f5(x, grad=False):
    return (x[0, 0] + 2 * x[1, 0] - 7) ** 2 + (2 * x[0, 0] + x[1, 0] - 5) ** 2


def f6(x, grad=False):
    if grad == False:
        return 100 * np.sqrt(np.abs(x[1, 0] - 0.01 * x[0, 0] ** 2)) + 0.01 * np.abs(x[0, 0] + 10)
    else:
        return 100 * torch.sqrt(torch.abs(x[1, 0] - 0.01 * x[0, 0] ** 2)) + 0.01 * torch.abs(x[0, 0] + 10)


def f7(x, grad=False):
    return 0.26 * (x[0, 0] ** 2 + x[1, 0] ** 2) - 0.48 * x[0, 0] * x[1, 0]


def f8(x, grad=False):
    if grad == False:
        return np.sin(3 * np.pi * x[0, 0]) ** 2 + (x[0, 0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[0, 0]) ** 2) + (
                x[1, 0] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1, 0]) ** 2)
    else:
        return torch.sin(3 * torch.pi * x[0, 0]) ** 2 + (x[0, 0] - 1) ** 2 * (
                1 + torch.sin(3 * torch.pi * x[0, 0]) ** 2) + (x[1, 0] - 1) ** 2 * (
                       1 + torch.sin(2 * torch.pi * x[1, 0]) ** 2)


def f9(x, grad=False):
    return 2 * x[0, 0] ** 2 - 1.05 * x[0, 0] ** 4 + x[0, 0] ** 6 / 6 + x[0, 0] * x[1, 0] + x[1, 0] ** 2


def f10(x, grad=False):
    if grad == False:
        return -np.cos(x[0, 0]) * np.cos(x[0, 0]) * np.exp(-((x[0, 0] - np.pi) ** 2 + (x[1, 0] - np.pi) ** 2))
    else:
        return -torch.cos(x[0, 0]) * torch.cos(x[0, 0]) * torch.exp(
            -((x[0, 0] - torch.pi) ** 2 + (x[1, 0] - torch.pi) ** 2))


def f11(x, grad=False):
    if grad == False:
        return -0.0001 * (np.abs(np.sin(x[0, 0]) * np.sin(x[1, 0]) * np.exp(
            np.abs(100 - np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2) / np.pi))) + 1) ** 0.1
    else:
        return -0.0001 * (torch.abs(torch.sin(x[0, 0]) * torch.sin(x[1, 0]) * torch.exp(
            torch.abs(10 - torch.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2) / torch.pi))) + 1) ** 0.1


def f12(x, grad=False):
    if grad == False:
        return -(x[1, 0] + 47) * np.sin(np.sqrt(np.abs(x[1, 0] + x[0, 0] / 2 + 47))) - x[0, 0] * np.sin(
            np.sqrt(np.abs(x[0, 0] - x[1, 0] - 47)))
    else:
        return -(x[1, 0] + 47) * torch.sin(torch.sqrt(torch.abs(x[1, 0] + x[0, 0] / 2 + 47))) - x[0, 0] * torch.sin(
            torch.sqrt(torch.abs(x[0, 0] - x[1, 0] - 47)))


if __name__ == '__main__':
    f = f12
    trace = dogleg(f, np.mat([[2.0], [1.5]]))
    trace_d = double_dogleg(f, np.mat([[2.0], [1.5]]))
    x_tr = [[], []]
    y_tr = [[], []]
    z_tr = [[], []]
    for i in range(101):
        x_tr[0].append(trace[0][i][0, 0])
        y_tr[0].append(trace[0][i][1, 0])
        z_tr[0].append(trace[1][i])
        x_tr[1].append(trace_d[0][i][0, 0])
        y_tr[1].append(trace_d[0][i][1, 0])
        z_tr[1].append(trace_d[1][i])

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    x = np.linspace(-2, 8, 100)
    y = np.linspace(-2, 15, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.mat([[a], [b]])) for a, b in zip(X[i], Y[i])] for i in range(x.shape[0])])
    ax.plot(x_tr[0], y_tr[0], z_tr[0], 'y', linewidth='1.5', alpha=1)
    ax.plot(x_tr[1], y_tr[1], z_tr[1], 'r', linewidth='2', alpha=1)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()
