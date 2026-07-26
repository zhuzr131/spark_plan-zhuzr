import numpy as np
import tensorcircuit as tc

n = 3

def grover_oracle(c):
    c.toffoli(0, 2, 1)
    c.X(2)
    c.toffoli(0, 2, 1)
    c.X(0)
    c.X(1)
    c.X(4)
    c.toffoli(0, 2, 4)
    c.toffoli(1, 4, 3)
    c.toffoli(0, 2, 4)
    c.X(0)
    c.X(1)
    c.X(4)
    c.toffoli(0, 2, 1)
    c.X(2)
    c.toffoli(0, 2, 1)
    return c

def grover_reflection(c):
    for i in range(n):
        c.H(i)
        c.X(i)
    c.multicontrol(*range(n), unitary=tc.gates.z(), ctrl=[1 for _ in range(n - 1)])
    for i in range(n):
        c.X(i)
        c.H(i)
    return c

def grover_algorithm(R):
    c = tc.Circuit(n + 2)
    c.X(n)
    for i in range(n + 1):
        c.H(i)
    for i in range(R):
        c = grover_oracle(c)
        c = grover_reflection(c)
    return c

N = 2 ** n
M = 1  # 单个目标状态 |001> (q0=1,q1=0,q2=0)
theta = np.arcsin(2. * np.sqrt(M * (N - M)) / N)
R = round(np.arccos(np.sqrt(M / N)) / theta)

c = grover_algorithm(R)
print(c.perfect_sampling())