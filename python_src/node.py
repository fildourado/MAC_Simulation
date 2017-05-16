import numpy as np

class node(object):
    """ 
    node class
    """

    def __init__(self):
        self.num_successful_transmissions = 0

    def print_num_transmissions(self):
        print self.num_successful_transmissions
