import random

import utils_ec as utils
import math


def explore(epsilon):
    rando = random.random() # random number between 0 and 1
    if rando < epsilon:
        return True
    else:
        return False


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
        # Create table for frequencies
        self.N = utils.create_q_table()
        #Save previous number of bounces
        self.bounces = 0
        #ADJUST THESE
        self.C = 2
        self.learning_rate = self.C
        self.discount_factor = 0.075
        #For Q-learning algorithm
        #Old state
        self.s = (0,0,0,0,0)
        #Old action
        self.a = 0
        #Old reward
        self.r = 0
        #Used in exploration function
        self.Ne = 10
        # number of iterations
        self.iterations = 0
        # for calculating average velocity in both the positive and negative directions
        self.x_accum_pos = 0
        self.x_pos_times = 0
        self.x_accum_neg = 0
        self.x_neg_times = 0

    def act(self, state, bounces, done, won):

        ball_x = 0
        ball_y = 0
        vel_x = 0
        vel_y = 0
        paddle_y = 0

        paddle_height = 0.2 #not sure if we can access this somehow

        #Discretize ball x and y into bins
        ball_x = int(math.floor(state[0] * self._x_bins)) - 1
        ball_y = int(math.floor(state[1] * self._y_bins)) - 1

        #Sometimes get 12 error for some reason?
        if ball_x > 11:
            ball_x = 11
        if ball_y > 11:
            ball_y = 11

        #Discretizing x-velocity ( x-velocity now has 4 states in it!)
        avg_vel_pos_x = 0.03596230868971933
        avg_vel_neg_x = -0.03633040902721528
        if (state[2] >= 0 and state[2] > avg_vel_pos_x):
            #very fast forward
            vel_x = 0
        elif (state[2] >= 0 and state[2] <= avg_vel_pos_x):
            #forward
            vel_x = 1
        elif (state[2] < 0 and state[2] < avg_vel_neg_x):
            #very fast backward
            vel_x = 2
        elif (state[2] < 0 and state[2] > avg_vel_neg_x):
            #backward
            vel_x = 3
        #Discretizing y-velocity
        if abs(state[3]) < 0.015:
            vel_y = 1
        elif state[3] > 0:
            vel_y = 2
        elif state[3] < 0:
            vel_y = 0

        #Discretize paddle location
        if state[4] == 1 - paddle_height:
            paddle_y = 11
        else:
            paddle_y = int(math.floor(self._paddle_locations * state[4]/(1 - paddle_height)))

        if self._train:

            #Calculate the reward
            reward = 0
            #If we bounced off the paddle, then reward = 1
            if done and won:
                reward = 2
            elif bounces > self.bounces:
                reward = 1
                self.bounces += 1
            #If we lost, reward = -1
            elif state[0] > 1:
                reward = -1

            #If anything else happened, 0

            #Check for terminal state
            if done:
                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.a + 1] = reward
                return 0
            else:
                #Increment frequency of previous state
                self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.a + 1] += 1

                #Find the max of differences between Q[s', a'] and Q[s, a] (ie which action will yield the most reward?)
                best_action = 0
                best_action_diff = 0
                action_found = False
                old_q = self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.a + 1]

                for action in self._actions:
                    diff = self.Q[ball_x, ball_y, vel_x, vel_y, paddle_y, action + 1] - old_q
                    if not action_found or best_action_diff < diff:
                        action_found = True
                        best_action_diff = diff
                        best_action = action

                #Update Q according to equation
                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.a + 1] += \
                        self.learning_rate * (self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.a + 1]) * (reward + self.discount_factor * best_action_diff)

            #Update Reward
            self.r = reward

        #Update states n' stuff
        self.s = (ball_x, ball_y, vel_x, vel_y, paddle_y)

        #Update previous action using basic exploration function defined on page 842 of the book
        #Find the action that maximizes the exploration function
        best_action = 0
        best_action_f = 0
        action_found = False

        possible_qs = self.Q[ball_x, ball_y, vel_x, vel_y, paddle_y]
        possible_ns = self.N[ball_x, ball_y, vel_x, vel_y, paddle_y]
        for action in self._actions:
            q_val = possible_qs[action + 1]
            if self._train:
                n_val = possible_ns[action + 1]
                if n_val < self.Ne:
                    #Since this is guaranteed to be the most optimistic, it's clearly largest
                    best_action = action
                    break
            if not action_found or best_action_f < q_val:
                best_action_f = q_val
                best_action = action
                action_found = True

        const = 10
        N = self.N[ball_x, ball_y, vel_x, vel_y, paddle_y, self.a + 1] #self.iterations
        epsilon = const / (const + N)
        self.a = random.randint(-1, 1) if self._train and explore(epsilon) else best_action

        self.iterations +=1

        # For computing the average forward and average backwards velocity in order to determine better bins
        if state[2] > 0:
            self.x_accum_pos += state[2]
            self.x_pos_times +=1
        else:
            self.x_accum_neg += state[2]
            self.x_neg_times +=1

        #Update the learning rate
        self.learning_rate = self.C / (self.C + self.N[ball_x, ball_y, vel_x, vel_y, paddle_y, self.a + 1])

        return self.a

    def train(self):
        self._train = True

    def eval(self):
        self._train = False
        avg_forward_vel = self.x_accum_pos / self.x_pos_times
        avg_backward_vel = self.x_accum_neg / self.x_neg_times
        print('avg_forward_val' + str(avg_forward_vel))
        print('avg_backward_val' + str(avg_backward_vel))
        print('breakpoint')


    def save_model(self, model_path):
        # At the end of training save the trained model
        utils.save(model_path,self.Q)

    def load_model(self, model_path):
        # Load the trained model for evaluation
        self.Q = utils.load(model_path)
