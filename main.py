from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

A = np.array([[10,2,0,0],
              [3,10,4,0],
              [0,1,7,5],
              [0,0,3,4]],dtype=float)

c = np.array([10,10,7,4], dtype=float)
a = np.array([0,3,1,3], dtype=float)
b = np.array([2,4,5,0], dtype=float)
f = np.array([3,4,5,6], dtype=float)

n = len(f)
p = n / 2
x = np.zeros(n, dtype=float)

coef_MPI = np.zeros(2, dtype=float)
xp_MPI = np.zeros(1, dtype=float)
x_MPI = np.zeros(p, dtype=float)

if rank == 0:

    # 1 step
    alpha = np.zeros(p + 1, dtype=float)
    betta = np.zeros(p + 1, dtype=float)

    alpha[1] = - b[0] / c[0]
    betta[1] = f[0] / c[0]

    for i in xrange(1, p):
        alpha[i + 1] = - b[i] / (a[i] * alpha[i] + c[i])
        betta[i + 1] = (f[i] - a[i] * betta[i]) / (a[i] * alpha[i] + c[i])

    # 2 step
    comm.Recv(coef_MPI, source=1, tag=2)

    # 3 step
    x[p - 1] = (alpha[p] * coef_MPI[0] + betta[p]) / (1 - alpha[p] * coef_MPI[1])
    xp_MPI[0] = x[p - 1]
    comm.Send(xp_MPI, dest=1, tag=3)

    # 4 step
    for i in reversed(xrange(0, p - 1)):
        x[i] = alpha[i + 1] * x[i + 1] + betta[i + 1]

    # 5 step
    comm.Recv(x_MPI, source=1, tag=5)
    for i in xrange(0, p):
        x[i + p] = x_MPI[i]

    # 6 step
    print "solved:"
    print x
    print "expected:"
    print np.linalg.solve(A, f)
elif rank == 1:

    # 1 step
    eps = np.zeros(n)
    nu = np.zeros(n)
    eps[n - 1] = - a[n - 1] / c[n - 1]
    nu[n - 1] = f[n - 1] / c[n - 1]

    for i in reversed(xrange(p - 1, n - 1)):
        eps[i] = - a[i] / (c[i] + b[i] * eps[i + 1])
        nu[i] = (f[i] - b[i] * nu[i + 1]) / (c[i] + b[i] * eps[i + 1])

    # 2 step
    coef_MPI[0] = nu[p]
    coef_MPI[1] = eps[p]
    comm.Send(coef_MPI, dest=0, tag=2)

    # 3 step
    comm.Recv(xp_MPI, source=0, tag=3)
    x[p - 1] = xp_MPI[0]

    # 4 step
    for i in xrange(p - 1, n - 1):
        x[i + 1] = eps[i + 1] * x[i] + nu[i + 1]

    # 5 step
    for i in xrange(0, p):
        x_MPI[i] = x[i + p]
    comm.Send(x_MPI, dest=0, tag=5)
