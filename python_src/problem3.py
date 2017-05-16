import numpy as np
import matplotlib.pyplot as plt
from node import node
from MAC_802_11 import MAC_802_11


"""
Simulation Parameters
"""

# probabilities of transmission and re-transmission
q = np.linspace(0.0001, 1.0, num=100)  # probability
#q = [0.5]
N = 12        # number of stations
H = 240.0     # Header bits
P = 1800.0    # Payload bits
R = 24.0      # Mbps
T = 15.0      # mini-slot duration microseconds
SIFS = 15.0   # short interframe spacing microseconds
DIFS = 60.0   # DCF interframe spacing microseconds
tau = 2       # propagation delay

SLOTS = 100000    # Simulation time microseconds
#SLOTS = 20
"""
Simulation
"""
protocol = MAC_802_11(p=0, tau=tau, N=N, H=H, P=P, R=R, T=T, SIFS=SIFS, DIFS=DIFS)
# iterate over the probability values

total_tx_count = []
# total_tx_collisions = []
# average_delay = []
# average_collisions = []
i = 0
for p in q:
    print i
    i += 1
    # reset the protocol sim
    protocol.reset(p=p, tau=tau, N=N, H=H, P=P, R=R, T=T, SIFS=SIFS, DIFS=DIFS)
    for slt in range(SLOTS):
        #print "TIME: %d" % (slt)
        # update the current channel state
        # if a node has finished transmitting in the previous slot then update the state
        protocol.update_channel_state(current_slt=slt)
        protocol.update_nodes(current_slt=slt)
        #print " "
    total_tx_count.append(protocol.total_tx_count)


total_time = SLOTS*T*(1e-6)
plt.plot(q, np.array(total_tx_count))
plt.show()
