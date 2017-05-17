import numpy as np
import sys
import matplotlib.pyplot as plt
from MAC_802_11 import MAC_802_11

"""
Simulation Parameters
"""

#SLOTS = 20


# probabilities of transmission and re-transmission
q = np.linspace(0.0001, 0.9, num=100)  # probability

N = 12      # number of stations
H = 240.0     # Header bits
P = 1800.0    # Payload bits
R = 24.0      # Mbps
sigma = 15.0  # mini-slot duration microseconds
SIFS = 15.0   # short interframe spacing microseconds
DIFS = 60.0   # DCF interframe spacing microseconds
tau = 2.0     # propagation delay
ACK = 112.0   # bits

Wmin = 4.0
rmax = 3.0

Ts = DIFS + H/R + P/R + tau + SIFS + ACK/R + tau
Tc = DIFS + H/R + P/R + tau
T = P/R + H/R      # data transmission time

S = []
SLOTS = 100000    # Simulation time microseconds


"""
Problem 1 Part C
"""

for p in q:

    Ptr = 1.0 - ((1.0 - p) ** N)
    #Pss_capture = (1.0*N*p)*(1.0-p)*(((1.0-p)**(N-2)) + (1.0 - ((1.0-p)**(N-2))))
    Pss_capture = (1.0*N*p)*((1.0-p)**(N-1))

    s = (T*Pss_capture) / ( (Ts * Pss_capture) + (Tc*Ptr)*(1.0 - (Pss_capture/Ptr)) + (sigma*((1.0-p)**N)) )

    S.append(s)

G = (N * q * T) / sigma
#G = N*q
S = np.array(S)

plt.figure(1)
plt.plot(G,S)
plt.xlim([0,7])
plt.ylim([0,1])
plt.title("Throughput vs Offered Load (N = 12)")
plt.xlabel("Offered Load (G)")
plt.ylabel("Throughput (S)")
plt.grid()

plt.savefig("S_vs_G_analytical.png")
#plt.show()


"""
Problem 1 Part D
"""
idx_max_s = np.argmax(S)
print S[idx_max_s]
print G[idx_max_s]


"""
Problem 2 Part B
"""

Wmin_t = 1.0 * Wmin * sigma

E_D = []
#q = [0.8]
for p in q:

    Ptr = 1.0 - ((1.0 - p) ** N)
    P_i = 1.0 - Ptr
    #Ps = N * p * ((1.0-p)**(N-1))
    Ps = (1.0-p)**N
    P_c = 1.0 - Ps - P_i

    theta = (sigma * P_i) + (Ts * Ps) + (Tc * P_c)

    term1 = (Wmin_t*theta) / (2.0*(1.0-((1.0-Ps)**(rmax+1.0))))
    #print "term 1:"
    #print theta
    #print (Wmin_t * theta)
    #print (2.0*(1.0-((1.0-Ps)**(rmax+1.0))))
    #print term1

    term2 = Ps * ( ( ( 1.0 - ( ( 2.0 * ( 1.0 - Ps ) )**(rmax + 1.0) ) ) ) / ( 1.0 - (2.0*(1.0-Ps) ) ) )
    #print "term 2:"
    #print term2

    term3 = 1.0 + ( (1.0-Ps)**(rmax + 1.0) )
    #print "term 3:"
    #print term3

    term4 = Tc * ((1.0-Ps)/Ps)
    #print "term 4:"
    #print term4

    term5 = ( ( (1.0 - Ps)**(rmax) ) * ( ( -1.0*Ps*rmax ) - 1.0 ) ) + 1.0
    #print "term 5:"
    #print term5

    term6 = 1.0 - ((1.0-Ps)**(rmax+1.0))
    #print "term 6:"
    #print term6

    D = Ts + (term1 * ( term2 - term3)) + (term4 * (term5 / term6))
    #print "D:"
    #print D

    E_D.append(D)

E_D = np.array(E_D)
E_D = E_D/sigma


plt.figure(2)
plt.plot(S,E_D)
plt.ylim([0,300])

plt.title("Average Delay vs Throughput (N = 12)")
plt.xlabel("Throughput (S)")
plt.ylabel("Average Delay (D) Mini-Slots")
plt.grid()

plt.savefig("D_vs_S_analytical.png")

#sys.exit()

"""
Problem 3 Simulation
"""
protocol = MAC_802_11(p=0, tau=tau, N=N, H=H, P=P, R=R, T=sigma, SIFS=SIFS, DIFS=DIFS)
# iterate over the probability values

total_tx_count = []
# total_tx_collisions = []
average_delay = []
# average_collisions = []
i = 0
for p in q:
    print i
    i += 1
    # reset the protocol sim
    protocol.reset(p=p, tau=tau, N=N, H=H, P=P, R=R, T=sigma, SIFS=SIFS, DIFS=DIFS)
    for slt in range(SLOTS):
        #print "TIME: %d" % (slt)
        # update the current channel state
        # if a node has finished transmitting in the previous slot then update the state
        protocol.update_channel_state(current_slt=slt)
        protocol.update_nodes(current_slt=slt)
        #print " "
    total_tx_count.append(protocol.total_tx_count)
    average_delay.append(protocol.delay_tracker)


# average mini-slot delays per packet
average_delay = np.array(average_delay)/np.array(total_tx_count)

total_time = 2*SLOTS*sigma*(1e-6)
max_packets_per_sec = (R*(1e6))/(P+H)

S_sim = np.array(total_tx_count)/total_time
S_sim = S_sim / max_packets_per_sec

plt.figure(3)
plt.plot(G,S_sim, label='Simulation')
plt.plot(G,S, label = 'Theoretical')
plt.xlim([0,7])
plt.ylim([0,1])
plt.title("Throughput vs Offered Load (N = 12)")
plt.xlabel("Offered Load (G)")
plt.ylabel("Throughput (S)")
plt.grid()
plt.legend()
plt.savefig('S_vs_G_sim.png')

plt.figure(4)
plt.plot(S, 2*average_delay, label='Simulation')
plt.plot(S, E_D, label = 'Theoretical')
plt.ylim([0,300])

plt.title("Average Delay vs Throughput (N = 12)")
plt.xlabel("Throughput (S)")
plt.ylabel("Average Delay (D) Mini-Slots")
plt.grid()
plt.legend()
plt.savefig('D_vs_S_sim.png')

#plt.show()