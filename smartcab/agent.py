import random
import numpy
from math import floor
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

## Reference https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.action_table = [None, 'forward', 'left', 'right'] ## List of all possible actions

        ## Decay function and assisting variables for GLIE
        ## SET - epsilon, decay rate
        self.epsilon = 0.2 ## Randomly take action X% of the time
        self.decay_rate = 0.005 ## Reduce epsilon by Y% after every iteration
        self.decay_factor = 1-self.decay_rate ## Generate decay factor (retained percentage of random choices)


        ## Q learning constants
        ## SET - gamma, alpha, defaultQ
        self.gamma = 0.5 ## Discount factor
        self.alpha = 0.5 ## set learning rate
        self.defaultQ = 0.01 ## default value for Q if none exists for given state


        ## Instantiate tables to tracking information later
        self.qtable = [] ## entire Q table
        self.t_table = [] ## used for reference in Update() method. Stores number of iterations per trial



    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required



        ## instantiate total reward summation
        self.total_reward = 0


        ## A probability table to use to decay random restarts
        self.probability_table = ['Use_q'] * int(floor((1*1000 - self.epsilon * 1000))) + ['random_restart'] * int(floor((self.epsilon * 1000)))
                
        ## Set new probability of taking random action
        self.epsilon = self.epsilon * self.decay_factor


   
    def find_q_action(self, state):

        ## finds the maximum Q given a state and returns the action with max Q
        Q_dict = self.findQ_no_action(state)
        if Q_dict != []:
            seq = [x['Q'] for x in Q_dict]
            Q_max = max(seq)
            Q_max_dict = filter(lambda x: x['Q'] == Q_max, Q_dict)
            action = Q_max_dict[0]['action']
            return action
        else:
            action = random.choice(self.action_table)
            return action


    def findQ(self, state, action):
        ## Search the Q table for the state including which action was taken
        Q = next((item for item in self.qtable if item['light'] == state['light'] and item['oncoming'] == state['oncoming'] and item['right'] == state['right'] and item['left'] == state['left'] and item['waypoint'] == state['waypoint'] and item['action'] == action), None)
        return Q

    def findQ_no_action(self, state):
        ## Search the Q table for the state without the action
        ## useful for deciding which action to take and finding Q(s', a')

        q = filter(lambda spec_states: spec_states['light'] == state['light'] and spec_states['oncoming'] == state['oncoming'] and spec_states['right'] == state['right'] and spec_states['left'] == state['left'] and spec_states['waypoint'] == state['waypoint'], self.qtable)

        return q

    def updateQ(self, state, action, reward, current_state):

        ## Set Q-learning equation variables
        s_prime = current_state ## more descriptive
        
        ## set these to avoid confusion with similarly named variables outside of method
        s = state
        a = action
        r = reward

        ## find Q and check existence
        Q = self.findQ(s, a)
        Q_exists = Q

        ## find Q' list of dictionaries
        Q_prime = self.findQ_no_action(s_prime)

        ## check if list is empty if true, set to default Q value
        if Q_prime == []:
            Q_prime_max_exists = False
            Q_prime_max = self.defaultQ
        
        ## if Q exists
        else:
            ## create a squence of Q_values
            seq = [x['Q'] for x in Q_prime]

            ## find max Q value
            Q_prime_max = max(seq)

            ## filter through dictionaries to find entire dictionary with given Q value
            Q_prime_max_dict = filter(lambda x: x['Q'] == Q_prime_max, Q_prime)

            ## confirm existence of max(Q(s',a'))
            Q_prime_max_exists = True


        ## check for existence of Q value before running Q Learning equation
        if Q == None:
            ## if Q does not exist, set to defaultQ value
            new_Q = self.defaultQ
        else:
            ## if Q does exist, set equal to new placeholder value for use in equation
            new_Q = Q['Q']


        ## Run the Q Learning equation with Learning Rate Alpha
        new_Q = (1-self.alpha) * new_Q + self.alpha * (r + self.gamma * Q_prime_max)

        ## Update the Q table
        if Q_exists == None:
        ## Insert a new entry to Q table if Q did not exist before
            Q_entry = {
                'light':            s['light'],
                'oncoming':         s['oncoming'],
                'right':            s['right'],
                'left':             s['left'],
                'waypoint':         s['waypoint'],
                'action':           a,
                'Q':                new_Q,
            }
            self.qtable.append(Q_entry)

        else:
            ## If Q did exist, replace existing Q with recalculated Q
            Q['Q'] = new_Q 


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        # Update state
        self.state = {'light': inputs['light'],'oncoming': inputs['oncoming'],'right': inputs['right'],'left': inputs['left'], 'waypoint': self.next_waypoint}
        # Select action according to your policy

        ## EXERCISE 1: initial action based on random choice of the action_table for exercise 1,
        ## will be commented out after the first questions are answered
        # action = random.choice(self.action_table)

        ## Create a decaying epsilon for a GLIE Q-Learning implementation
        ## Tune decay rate and epsilon in __init__
        action_selector = random.choice(self.probability_table)



        ## TODO: Set up Q_hat equation
        ## TODO: Exercise 2:

        ## selector to use the recommended action from Q table or take a random action
        ## reference probability_table
        if action_selector == 'Use_q':
            action = self.find_q_action(self.state)
        else:
            action = random.choice(self.action_table)

        

        # Execute action and get reward
        reward = self.env.act(self, action)


        ## save total reward for reference
        self.total_reward = self.total_reward + reward

        ## sense new environment after move
        self.next_inputs = self.env.sense(self)


        ## define next state (s') after move
        self.next_state = {'light': self.next_inputs['light'],'oncoming': self.next_inputs['oncoming'],'right': self.next_inputs['right'],'left': self.next_inputs['left'], 'waypoint': self.next_waypoint}

        
        ## run the Q learning algorithm based in the updateQ function
        self.updateQ(self.state, action, reward, self.next_state)



        # TODO: Learn policy based on state, action, reward
        # Print information about the algorithm
        if self.env.done:

            ## print to show entire Q table after each trial
            # print self.qtable


            ## print to show how many state action pairs have been discovered
            print "\n \nQ-Table Length: " + str(len(self.qtable)) + " state-action pairs \n \n"


            ## Track number of moves to destination by trial by adding t to a table reset at beginning of algorithm
            self.t_table.append(t)


            ## print the number of moves table. A quick way to see if the number of moves is decreasing as the trials proceed
            print "Moves to destination"
            print self.t_table

            print "Total Times reached destination within time"
            print len(self.t_table)


            ## print the average number of moves to the destination, should decrease as number of trials increases
            print "Average moves"
            print sum(self.t_table)/len(self.t_table)



            ## Total reward of each trial. Helpful in identifying if the algo gets stuck in a non-productive
            ## action loop by collecting a net positive reward without advancing towards the goal
            print "Total Iteration reward"
            print self.total_reward

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.000005)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
