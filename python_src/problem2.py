import numpy as np
import matplotlib.pyplot as plt

"""
Part C
"""
# probabilities of transmission and re-transmission
q = np.linspace(0.000000001, 1.0, num=10000)  # probability

N = 12.0        # number of stations
H = 240.0     # Header bits
P = 1800.0    # Payload bits
R = 24.0      # Mbps
sigma = 15.0      # mini-slot duration microseconds
SIFS = 15.0   # short interframe spacing microseconds
DIFS = 60.0   # DCF interframe spacing microseconds
tau = 2.0       # propagation delay
ACK = 100.0     # bits
Wmin = 4
rmax = 3
Ts = DIFS + H/R + P/R + tau + SIFS + ACK/R + tau
Tc = DIFS + H/R + P/R + tau
T = P/R + H/R      # data transmission time
