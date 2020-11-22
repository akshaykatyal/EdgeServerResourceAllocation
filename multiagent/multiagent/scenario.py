import numpy as np
#this is the basic scenario class

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world as given in the scenario class
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
