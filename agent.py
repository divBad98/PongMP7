import utils

statesToFreqs = {} #may need this for learning, but am having trouble working it in
statesToResults = {} #may need this for learning, but am having trouble working it in

class Agent:
    def __init__(self, actions):
        self._actions = actions
        self._train = True
        self._x_bins = utils.X_BINS
        self._y_bins = utils.Y_BINS
        self._v_x = utils.V_X
        self._v_y = utils.V_Y
        self._paddle_locations = utils.PADDLE_LOCATIONS
        self._num_actions = utils.NUM_ACTIONS
        # Create the Q Table to work with
        self.Q = utils.create_q_table()

    def act(self, state, bounces, done, won):
        #TODO - fill out this function
        ret = 0
        stateTuple = tuple(state)
        
        #Discretizing x-velocity
        if (state[2] > 0):
            state[2] = 1
        elif (state[2] < 0):
            state[2] = -1
        #Discretizing y-velocity
        if (state[3] > 0):
            if (state[3] < 0.015):
                state[3] = 0
            else:
                state[3] = 1
        elif (state[3] < 0):
            state[3] = -1

        if (state[1] < state[4]):
            #Up case
            if stateTuple in statesToResults:
                statesToResults[stateTuple][2] += 1
            else:
                statesToResults[stateTuple] = [0, 0, 1]
            ret = self._actions[0]
        elif (state[1] > state[4]):
            #Down case
            if stateTuple in statesToResults:
                statesToResults[stateTuple][0] += 1
            else:
                statesToResults[stateTuple] = [1, 0, 0]
            ret = self._actions[2]
        else:
            #No change case
            if stateTuple in statesToResults:
                statesToResults[stateTuple][1] += 1
            else:
                statesToResults[stateTuple] = [0, 1, 0]
            ret = self._actions[1]
        if stateTuple in statesToFreqs:
            statesToFreqs[stateTuple] += 1
        else:
            statesToFreqs[stateTuple] = 1
        return ret

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    def save_model(self, model_path):
        # At the end of training save the trained model
        utils.save(model_path,self.Q)

    def load_model(self, model_path):
        # Load the trained model for evaluation
        self.Q = utils.load(model_path)
