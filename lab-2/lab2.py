from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split


class Segment(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def length(self):
        return self.b - self.a


def function(t, x1, x2):
    return 2 * (x1.a - t * x2.a) * (x1.a - t * x2.a) + (x1.a - t * x2.a) * (x1.b - t * x2.b) + (x1.b - t * x2.b) * (
            x1.b - t * x2.b)


def func(x):
    return 2 * x.a * x.a + x.a * x.b + x.b * x.b


def grad_func(x):
    return Segment(4 * x.a + x.b, x.a + 2 * x.b)


def numberOfFibonacci(n):
    if n < 1:
        return 0
    if n == 1 or n == 2:
        return 1
    else:
        num1 = 1
        num2 = 1
        for i in range(3, n):
            temp = num1 + num2
            num1 = num2
            num2 = temp
        return num2


def goldenRatio(startSeg, e, f1, f2, x1, x2):
    f = 0.5 * (1 + sqrt(5))
    seg = startSeg
    if seg.length() < e:
        return 0.5 * (seg.a + seg.b)
    else:
        a = seg.b - (seg.b - seg.a) / f
        b = seg.a + (seg.b - seg.a) / f
        fx1 = f1 if f1 is not None else function(a, x1, x2)
        fx2 = f2 if f2 is not None else function(b, x1, x2)
        if fx1 >= fx2:
            seg.a = a
            return goldenRatio(seg, e, fx2, None, x1, x2)
        else:
            seg.b = b
            return goldenRatio(seg, e, None, fx1, x1, x2)


def fibonacci(startSeg, iter, x1, x2):
    seg = startSeg
    count = 0
    a = seg.a + (seg.b - seg.a) * (numberOfFibonacci(iter - 2) / numberOfFibonacci(iter))
    b = seg.a + (seg.b - seg.a) * (numberOfFibonacci(iter - 1) / numberOfFibonacci(iter))
    f1 = function(a, x1, x2)
    f2 = function(b, x1, x2)
    count += 2
    iter -= 1
    while iter > 1:
        if f1 > f2:
            seg.a = a
            a = b
            b = seg.b - (a - seg.a)
            f1 = f2
            f2 = function(b, x1, x2)
            count += 1
        else:
            seg.b = b
            b = a
            a = seg.a + (seg.b - b)
            f2 = f1
            f1 = function(a, x1, x2)
            count += 1
        iter -= 1
    return (a + b) / 2


def find_x(number, x, grad_meaning, f_history, eps2, t):
    if number == 1:
        t = 0.03
        x.a = -t * grad_meaning.a + x.a
        x.b = -t * grad_meaning.b + x.b
    if number == 2:
        stop = 0
        while stop == 0:
            x1 = Segment(-t * grad_meaning.a + x.a, -t * grad_meaning.b + x.b)
            qwerty = f_history[-1] - eps2 * t * sqrt(
                grad_meaning.a * grad_meaning.a + grad_meaning.b * grad_meaning.b) * sqrt(
                grad_meaning.a * grad_meaning.a + grad_meaning.b * grad_meaning.b)
            if func(x1) <= qwerty:
                stop = 1
                x = x1
            else:
                t /= 2
    if number == 3:
        t = goldenRatio(Segment(0, 0.2), 0.1, None, None, Segment(x.a, x.b), grad_meaning)
        x.a = -t * grad_meaning.a + x.a
        x.b = -t * grad_meaning.b + x.b
    if number == 4:
        t = fibonacci(Segment(0, 0.2), 7, Segment(x.a, x.b), grad_meaning)
        x.a = -t * grad_meaning.a + x.a
        x.b = -t * grad_meaning.b + x.b
    print(t, " - t")
    return t, x


def gradient_descent(M, x0, eps1, eps2):
    x = Segment(x0.a, x0.b)
    x_history = np.array([x0])
    f_history = np.array([func(x)])
    k = 0
    check = 0
    t = 0.25
    while 1:
        k = k + 1
        grad_meaning = grad_func(x)
        if sqrt(grad_meaning.a * grad_meaning.a + grad_meaning.b * grad_meaning.b) < eps1:
            break
        if k >= M:
            break
        t, x = find_x(3, x, grad_meaning, f_history, eps2, t)
        x_history = np.vstack((x_history, Segment(x.a, x.b)))
        f_history = np.vstack((f_history, func(x)))
        if sqrt((x_history[-1][0].a - x_history[-2][0].a) * (x_history[-1][0].a - x_history[-2][0].a) + (
                x_history[-1][0].b - x_history[-2][0].b) * (x_history[-1][0].b - x_history[-2][0].b)) < eps2 and abs(
            f_history[-1] - f_history[-2]) < eps2:
            check = check + 1
            if check == 2:
                break
    print(k, " - k")
    print(x_history[-1][0].a, " - x1")
    print(x_history[-1][0].b, " - x2")
    print(f_history[-1][0], " - func")
    temp_x = np.zeros((1, 2))
    for item in x_history:
        temp = np.array([item[0].a, item[0].b])
        temp_x = np.vstack((temp_x, temp))
    return temp_x, f_history


def conjugate_gradient(e, x0, N):
    x = Segment(x0.a, x0.b)
    x_history = np.array([x0])
    f_history = np.array([func(x)])
    grad_meaning = grad_func(x)
    k = 0
    while 1:
        n = 0
        p = grad_meaning
        while sqrt(grad_meaning.a * grad_meaning.a + grad_meaning.b * grad_meaning.b) >= e:
            k = k + 1
            t = goldenRatio(Segment(0, 1), e, None, None, Segment(x.a, x.b), Segment(p.a, p.b))
            x.a = -t * p.a + x.a
            x.b = -t * p.b + x.b
            x_history = np.vstack((x_history, Segment(x.a, x.b)))
            f_history = np.vstack((f_history, func(x)))
            grad_meaning_k = grad_meaning
            grad_meaning = grad_func(x)
            if n + 1 == N:
                break
            b = (grad_meaning.a * grad_meaning.a + grad_meaning.b * grad_meaning.b) / (
                    grad_meaning_k.a * grad_meaning_k.a + grad_meaning_k.b * grad_meaning_k.b)
            p.a = b * (-p.a) + grad_meaning.a
            p.b = b * (-p.b) + grad_meaning.b
            n += 1
        if sqrt(grad_meaning.a * grad_meaning.a + grad_meaning.b * grad_meaning.b) < e:
            break;
        print(k, " - k")
        print(x_history[-1][0].a, " - x1")
        print(x_history[-1][0].b, " - x2")
        print(f_history[-1][0], " - func")
    print(k, " - k")
    print(x_history[-1][0].a, " - x1")
    print(x_history[-1][0].b, " - x2")
    print(f_history[-1][0], " - func")
    temp_x = np.zeros((1, 2))
    for item in x_history:
        temp = np.array([item[0].a, item[0].b])
        temp_x = np.vstack((temp_x, temp))
    return temp_x, f_history


def visualize_fw():
    xcoord = np.linspace(-10.0, 10.0, 50)
    ycoord = np.linspace(-10.0, 10.0, 50)
    w1, w2 = np.meshgrid(xcoord, ycoord)
    pts = np.vstack((w1.flatten(), w2.flatten()))
    pts = pts.transpose()
    f_vals = np.sum(pts * pts, axis=1)
    function_plot(pts, f_vals)
    plt.title('Objective Function Shown in Color')
    plt.show()
    return pts, f_vals



def annotate_pt(text, xy, xytext, color):
    plt.plot(xy[0], xy[1], marker='P', markersize=10, c=color)
    plt.annotate(text, xy=xy, xytext=xytext,
                 # color=color,
                 arrowprops=dict(arrowstyle="->",
                                 color=color,
                                 connectionstyle='arc3'))



def function_plot(pts, f_val):
    f_plot = plt.scatter(pts[:, 0], pts[:, 1],
                         c=f_val, vmin=min(f_val), vmax=max(f_val),
                         cmap='RdBu_r')
    plt.colorbar(f_plot)
    annotate_pt('global minimum', (0, 0), (-5, -7), 'yellow')


pts, f_vals = visualize_fw()


def visualize_learning(w_history):
    function_plot(pts, f_vals)
    plt.plot(w_history[:, 0], w_history[:, 1], marker='o', c='magenta')
    annotate_pt('minimum found',
                (w_history[-1, 0], w_history[-1, 1]),
                (-1, 7), 'green')
    iter = w_history.shape[0]
    for w, i in zip(w_history, range(iter - 1)):
        plt.annotate("",
                     xy=w, xycoords='data',
                     xytext=w_history[i + 1, :], textcoords='data',
                     arrowprops=dict(arrowstyle='<-',
                                     connectionstyle='angle3'))


def solve_fw():
    rand = np.random.RandomState(19)
    w_init = rand.uniform(-10, 10, 2)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(18, 12))
    learning_rates = [0.05, 0.2, 0.5, 0.8]
    momentum = [0, 0.5, 0.9]
    ind = 1

    for alpha in momentum:
        for eta, col in zip(learning_rates, [0, 1, 2, 3]):
            plt.subplot(3, 4, ind)
            w_history, f_history = gradient_descent(100, Segment(-5, 5), 0.1, 0.15)
            #w_history, f_history = conjugate_gradient(0.1, Segment(-5, 5), 5)

            visualize_learning(w_history)
            ind = ind + 1
            plt.text(-9, 12, 'Learning Rate = ' + str(eta), fontsize=13)
            if col == 1:
                plt.text(10, 15, 'momentum = ' + str(alpha), fontsize=20)

    fig.subplots_adjust(hspace=0.5, wspace=.3)
    plt.show()


solve_fw()
gradient_descent(100, Segment(0.5, 1), 0.001, 0.0015)
conjugate_gradient(0.1, Segment(0.5, 1), 2)
