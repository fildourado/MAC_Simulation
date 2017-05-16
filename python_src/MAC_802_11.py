import numpy as np
from node import node

class MAC_802_11(object):

    """ 
    802.11 simulation class
    """
    def __init__(self, p, tau, N, H, P, R, T, SIFS, DIFS):
        print "Initializing 802.11 MAC Simulation"
        self.reset(p=p, tau=tau, N=N, H=H, P=P, R=R, T=T, SIFS=SIFS, DIFS=DIFS)

    def reset(self,p, tau, N, H, P, R, T, SIFS, DIFS):
        self.p = p          # probability of idle node deciding to tx in a slot
        self.tau = tau
        self.N = N          # number of stations or nodes
        self.H = H          # packet header size bits
        self.P = P          # packet payload size bits
        self.R = R          # data rate Mbps
        self.T = T          # time length of mini slot us
        self.SIFS = SIFS    # time length of the short inter frame spacing
        self.DIFS = DIFS    # time length of the delay inter frame spacing
        self.rmax = 3       # max number of retransmissions

        # useful manipulations of inputs
        self.DIFS_slt_cnt = np.int(np.ceil((1.0*self.DIFS) / self.T))
        self.packet_length_us = (1.0*(self.P + self.H)) / self.R
        self.packet_slt_cnt = np.int(np.ceil((1.0*( self.packet_length_us + self.SIFS + self.tau)) / self.T))
        #print self.DIFS_slt_cnt
        #print self.packet_slt_cnt

        # tracks the channel state so nodes can sense and do collision avoidance
        # 0 = no one transmitting
        # 1 = 1 node transmitting
        self.channel_state = 0

        # vector to track each nodes' state
        # 0 = idle
        # 1 = transmit ready, waiting for DIFS to elapse or waiting for backoff to elapse
        # 2 = transmitting
        # 3 = waiting to re-tx
        self.node_state = np.zeros(N)

        # tracks how many times a node as attempted to re-tx
        self.node_tx_count = np.zeros(N)

        # tracks the transmitting node expiration slot time
        self.tx_node_expiration = 0

        # default the schedule to -1
        self.node_tx_schedule = np.zeros(N) - np.ones(N)

        # tracks the total packet transmit count
        self.total_tx_count = 0

        self.current_txing_node = -1

        self.Wmin = 4

    def update_channel_state(self, current_slt):
        # update the channel state
        # if the channel was busy in the previous slot, check if node that was transmitting has finished
        if self.channel_state == 1:
            if current_slt == self.tx_node_expiration:
                self.channel_state = 0
                # makae the node that just finished reset to idle
                self.node_state[self.current_txing_node] = 0
                self.tx_node_expiration = -1
                self.total_tx_count += 1
                self.current_txing_node = -1
                #print "successful transmission"


    def update_nodes(self, current_slt):

        #print "node state:"
        #print self.node_state
        #print "node schedule:"
        #print self.node_tx_schedule
        #print "transmit/retx count:"
        #print self.node_tx_count
        # check if any node wants to transmit and update its state
        # for idle states, we update the tx state with probability p
        idx = np.where(self.node_state == 0)
        probs = np.random.uniform(0, 1, len(idx[0]))

        # someone is currently transmitting
        if self.channel_state == 1:
            #print "Channel is Busy"

            # add 1 to the tx schedule of nodes to simulate a wait if channel is sensed busy.
            # backoff countdowns only happen when channel is idle
            self.node_tx_schedule += 1

            # if the channel is busy when a packet arrives for transmission at an idle node,
            # then give it a random backoff
            prob_idx = np.where(probs < self.p)
            node_idx = idx[0][prob_idx[0]]
            self.node_state[node_idx] = 1
            self.node_tx_count[node_idx] = 0
            #self.node_state[idx] = np.where(probs > self.p, self.tx_state[idx], 3)

            #Wmax = (2 ** (self.node_tx_count[node_idx] - 1)) * self.Wmin
            #backoff = np.int32(np.random.uniform(1, Wmax, len(node_idx)))
            #backoff = np.int32(np.random.uniform(1, Wmax, self.N))

            self.node_tx_schedule[node_idx] = current_slt + self.DIFS_slt_cnt


        elif self.channel_state == 0:
            #print "Channel is Idle"

            # check nodes that need to transmit right now
            idx1 = np.where(self.node_state == 1)
            #nodes_to_tx_now = idx1[0]
            nodes_to_tx_now = np.where(self.node_tx_schedule[idx1[0]] == current_slt)
            nodes_to_tx_now = idx1[0][nodes_to_tx_now[0]]

            # check the re-txing nodes
            idx2 = np.where(self.node_state == 3)
            nodes_to_retx_now = np.where(self.node_tx_schedule[idx2[0]] == current_slt)
            nodes_to_retx_now = idx2[0][nodes_to_retx_now[0]]

            # get an array of nodes that need to tx now
            txing_nodes = np.concatenate((nodes_to_tx_now, nodes_to_retx_now))

            # check if there was a collision
            if len(txing_nodes) > 1:
                #print "Collision! These nodes:"
                #print txing_nodes

                # keep the channel idle as no one transmits
                self.channel_state = 0

                # bad luck, it was time to transmit and channel just went active so we cant wait DIFS to retx, we have to wait a random backoff
                prob_idx = np.where(probs < self.p)
                node_idx = idx[0][prob_idx[0]]
                #self.node_state[node_idx] = 3
                self.node_state[node_idx] = 1

                #self.node_tx_count[node_idx] = 1
                self.node_tx_count[node_idx] = 0
                #Wmax = (2 ** (self.node_tx_count[node_idx] - 1)) * self.Wmin
                ##print Wmax
                #backoff = np.int32(np.random.uniform(1, Wmax, len(node_idx)))
                self.node_tx_schedule[node_idx] = current_slt + self.DIFS_slt_cnt


                # there was a collision in this slot
                # update node states
                self.node_state[nodes_to_tx_now] = 3

                # of the nodes that were in a single tx state, reset the tx counter
                self.node_tx_count[nodes_to_tx_now] = 1

                # increase re tx count of nodes in re-tx state
                self.node_tx_count[nodes_to_retx_now] += 1

                # reset nodes that have timed out
                idx_of_timed_out_nodes = np.where( self.node_tx_count[txing_nodes] == (self.rmax + 1) )
                if len(idx_of_timed_out_nodes[0]) > 0:
                    # set timed out nodes back to idle
                    self.node_state[idx_of_timed_out_nodes[0]] = 0
                    self.node_tx_schedule[idx_of_timed_out_nodes[0]] = -1
                    self.node_tx_count[idx_of_timed_out_nodes[0]] = 0

                Wmax = (2**(self.node_tx_count - 1))*self.Wmin

                backoff = np.int32(np.random.uniform(1, Wmax, self.N))

                self.node_tx_schedule[txing_nodes] = current_slt + backoff[txing_nodes]


            elif len(txing_nodes) == 1:
                #print "One node needs to TX:"
                #print txing_nodes

                # channel is now busy
                self.channel_state = 1

                # no collision and a single node wants to tx!
                self.current_txing_node = txing_nodes[0]
                self.tx_node_expiration = current_slt + self.packet_slt_cnt - 1
                self.node_state[txing_nodes[0]] = 2

                # bad luck, it was time to transmit and channel just went active so we cant wait DIFS to retx,
                # we have to wait a random backoff
                prob_idx = np.where(probs < self.p)
                node_idx = idx[0][prob_idx[0]]
                #self.node_state[node_idx] = 3
                self.node_state[node_idx] = 1
                self.node_tx_count[node_idx] = 0
                #Wmax = (2 ** (self.node_tx_count[node_idx] - 1)) * self.Wmin
                #backoff = np.int32(np.random.uniform(1, Wmax, self.N))
                #backoff = np.int32(np.random.uniform(1, Wmax, len(node_idx)))
                self.node_tx_schedule[node_idx] = current_slt + self.DIFS_slt_cnt

            else:
                # keep the channel idle
                self.channel_state = 0

                #if len(idx) > 0:
                #    print "Channel idle and new nodes want to tx, wait DIFS"
                #else:
                #    print "Channel idle, NO new nodes want to tx"
                # no one transmitted and channel was idle
                # if any nodes need to transmit, make them wait DIFS
                prob_idx = np.where(probs < self.p)
                node_idx = idx[0][prob_idx[0]]
                self.node_state[node_idx] = 1
                self.node_tx_count[node_idx] = 0
                self.node_tx_schedule[node_idx] = current_slt + self.DIFS_slt_cnt

        else:
            print "Channel Error"


