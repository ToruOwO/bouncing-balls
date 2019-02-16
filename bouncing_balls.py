"""
This script is modified based on the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

import argparse
import _pickle as pickle
import pdb

import h5py
import matplotlib
import scipy.io
from numpy import *
from scipy import *
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

shape_std = shape
r_list = None
m_list = None


def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()


size_std = size


def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()


det = linalg.det


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x):
    return sqrt((x ** 2).sum())


def sigmoid(x):
    return 1. / (1. + exp(-x))


SIZE = 10


# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2):
    # X records (x,y) coordinates of n balls in T time steps
    # r should be rather small
    X = zeros((T, n, 2), dtype='float')
    v = randn(n, 2)
    v = v / norm(v) * .5

    # generate initial configuration
    good_config = False
    while not good_config:
        x = 2 + rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r_list[i] < 0:
                    good_config = False
                if x[i][z] + r_list[i] > SIZE:
                    good_config = False

        # check if any two balls overlap
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r_list[i] + r_list[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # run simulation for T steps

        for i in range(n):
            X[t, i] = x[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r_list[i] < 0:  v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r_list[i] > SIZE: v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r_list[i] + r_list[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m_list[i], m_list[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def matricize(X, res):
    T, n = shape(X)[0:2]

    A = zeros((T, res, res), dtype='float')

    # create a rectangular grid out of an array of x values and an array of y values.
    [I, J] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t] += exp(-(((I - X[t, i, 0]) ** 2 + (J - X[t, i, 1]) ** 2) / (r_list[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128):
    x = bounce_n(T, n);
    A = matricize(x, res)
    return A


def bounce_vec(res, n=2, T=128):
    x = bounce_n(T, n);
    V = matricize(x, res)
    return V


def show_sample(V, logdir, length, animate=True):
    T = len(V)
    res = int(sqrt(shape(V)[1]))

    images = []

    # save static frames as PNG images
    for t in range(length):
        plt.imshow(V[t].reshape(res, res), cmap=matplotlib.cm.Greys_r)
        # Save it
        fname = logdir + '/' + str(t) + '.png'
        plt.savefig(fname)

        images.append(Image.open(fname))

    # generate GIF from the image files
    if animate:
        gif_name = 'sample_l{}.gif'.format(length)
        images[0].save(gif_name, format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)


# subprocess.run("ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='bouncing_balls')
    parser.add_argument('--log_dir', type=str, default='./sample')
    parser.add_argument('--num_balls', type=int, default=3)
    parser.add_argument('--radius', '-r', type=float, default=1.2, help='mean radius of bouncing balls')
    parser.add_argument('--resolution', '-res', type=int, default=64, help='resolution')
    parser.add_argument('-T', type=int, default=51, help='video sequence length')
    parser.add_argument('-N', type=int, default=1000, help='number of data samples')
    parser.add_argument('--sample_length', type=int, default=50, help='length of test video')
    parser.add_argument('--test', default=False, action='store_true', help='run test script only if True')

    args = parser.parse_args()

    res = args.resolution
    T = args.T
    N = args.N
    n = args.num_balls

    # list of radii of all balls
    # r_list = [args.radius] * n
    r_list = [0.5, 1, 2]

    # list of weights of all balls
    # m_list = [(i+1)*2 for i in range(n)]
    m_list = [1, 10, 50]


    def gen_train_data():
        print("Generating training data...")

        dat = empty((N, T, res, res))
        for i in range(N):
            dat[i] = bounce_vec(res=res, n=n, T=T)

        # save as h5py
        with h5py.File('{}_training_data.h5'.format(args.data_name), 'w') as hf:
            hf.create_dataset('{}_training_data'.format(args.data_name), data=dat)


    def gen_test_data():
        print("Generating test data...")

        # testing data size
        N = args.N // 20

        dat = empty((N, T, res, res))
        for i in range(N):
            dat[i] = bounce_vec(res=res, n=n, T=T)

        # save as h5py
        with h5py.File('{}_testing_data.h5'.format(args.data_name), 'w') as hf:
            hf.create_dataset('{}_testing_data'.format(args.data_name), data=dat)


    def gen_test_video():
        print("Generating test video sequence...")
        sample = bounce_vec(res=res, n=n, T=T)

        # show one video
        show_sample(sample, args.log_dir, args.sample_length)


    if args.test:
        gen_test_video()
    else:
        gen_train_data()
        gen_test_data()
