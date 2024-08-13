from typing import Set
#from ray.rllib.env.multi_agent_env import MultiAgentEnv
import copy
import gymnasium as gym
import numpy as np
#from ray.rllib.utils.typing import AgentID
from scipy.stats import poisson, randint
from utils import get_stage, get_retailers, create_network, create_adjacency_matrix, find_connections


"""
This environment is  a Multi-Product, Multi-Echelon Inventory Management Environment for a Multi-Agent System

Agent for each node and each product 
Action Space - order replenishment between two nodes. 
Observation Space - Inventory position, backlog and demand at each node
Reward - Total profit over the entire system 

Assumption: Max order or invenory capacity is based on per node per product; not total per node 

"""
class MultiAgentInvManagementDiv():
    def __init__(self, config):

        self.config = config.copy()
        # Number of Periods in Episode
        self.num_periods = config.get("num_periods", 365)

        # Structure
        self.independent = config.get("independent", True)
        self.share_network = config.get("share_network", False)
        self.num_nodes = config.get("num_nodes", 5)
        self.num_products = config.get("num_products", 10)
        self.node_names = []
        for i in range(self.num_nodes):
            for p in range(self.num_products):
                node_name = "node_" + str(i) + str(p)
                self.node_names.append(node_name)

        self.connections = config.get("connections",{0: [1], 1: [2,3], 2: [4], 3:[], 4:[]})
        self.network = create_network(self.connections)
        self.adjacency = create_adjacency_matrix(self.connections, self.num_nodes, self.num_products)
        self.order_network = np.transpose(self.network)
        self.retailers = get_retailers(self.network)
        self.non_retailers = list()
        for i in range(self.num_nodes):
            if i not in self.retailers:
                self.non_retailers.append(i)
        self.upstream_node = dict()
        for i in range(1, self.num_nodes):
            self.upstream_node[i] = np.where(self.order_network[i] == 1)[0][0]

        self.num_stages = get_stage(node=int(self.num_nodes - 1), network=self.network) + 1
        self.a = config.get("a", -1)
        self.b = config.get("b", 1)

        self.num_agents = config.get("num_agents", self.num_nodes * self.num_products)
        self.num_agents_type = config.get("num_agents_type", [1]* self.num_agents)
        self.num_types = config.get("num_types", self.num_nodes * self.num_products)
        self.inv_init = config.get("init_inv", np.ones((self.num_nodes, self.num_products))*100)
        self.inv_target = config.get("inv_target", np.ones((self.num_nodes, self.num_products)) * 15)
        self.delay = config.get("delay", np.ones((self.num_nodes, self.num_products), dtype=np.int32))
        self.time_dependency = config.get("time_dependency", False)
        self.prev_actions = config.get("prev_actions", True)
        self.prev_demand = config.get("prev_demand", True)
        self.prev_length = config.get("prev_length", 10)
        self.max_delay = np.max(self.delay)
        if self.max_delay == 0:
            self.time_dependency = False


        # Price of goods
        stage_price = np.arange(self.num_stages) + 2
        stage_cost = np.arange(self.num_stages) + 1
        self.node_price = np.zeros((self.num_nodes, self.num_products))
        self.node_cost = np.zeros((self.num_nodes, self.num_products))
        for i in range(self.num_nodes):
            for p in range(self.num_products):
                self.node_price[i][p] = stage_price[get_stage(i, self.network)] 
                self.node_cost[i][p] = stage_cost[get_stage(i, self.network)]

        self.price = config.get("price", np.flip(np.arange(self.num_stages + 1) + 1))

        # Stock Holding and Backlog cost
        self.stock_cost = config.get("stock_cost", np.ones((self.num_nodes, self.num_products))*0.5)
        self.backlog_cost = config.get("backlog_cost", np.ones((self.num_nodes, self.num_products)))
        
        # Customer demand
        self.demand_dist = config.get("demand_dist", "custom")
        self.SEED = config.get("seed", 52)
        np.random.seed(seed=int(self.SEED))

        # Customer demand noise
        self.noisy_demand = config.get("noisy_demand", False)
        self.noisy_demand_threshold = config.get("noisy_demand_threshold", 0)

        # Lead time noise
        self.noisy_delay = config.get("noisy_delay", True)
        self.noisy_delay_threshold = config.get("noisy_delay_threshold", 0.1)

        # Capacity
        self.inv_max = config.get("inv_max", np.ones((self.num_nodes, self.num_products), dtype=np.int16) * 100)
        order_max = np.zeros((self.num_nodes, self.num_products))

        for i in range(1, self.num_nodes):
            for p in range(self.num_products):
                indices = np.where(self.order_network[i] == 1)
                if indices[0].size > 0:
                    selected_index = indices[0][0]  # Select the first matching index
                    order_max[i][p] = self.inv_max[selected_index][p]
                else:
                    order_max[i][p] = 0  # Or some default value if no match is found
        order_max[0][:] = self.inv_max[0][:]
        self.order_max = config.get("order_max", order_max)

        # Number of downstream nodes of a given node
        self.num_downstream = dict()
        self.demand_max = copy.deepcopy(self.inv_max)
        for i in range(self.num_nodes):
            self.num_downstream[i] = np.sum(self.network[i])
            downstream_max_demand = np.zeros(self.num_products)
            for j in range(len(self.network[i])):
                if self.network[i][j] == 1:
                    downstream_max_demand += self.order_max[j]
            for p in range(self.num_products):
                if downstream_max_demand[p] > self.demand_max[i][p]:
                    self.demand_max[i][p] = downstream_max_demand[p]

        self.done = set()

        self.action_space = gym.spaces.Box(
            low=np.ones(1)*self.a,
            high=np.ones(1)*self.b,
            dtype=np.float64,
            shape=(1,)
        )


        # observation space (Inventory position at each echelon, which is any integer value)
        if not self.share_network:
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.max_delay, dtype=np.float64)*self.a,
                    high=np.ones(3 + self.max_delay, dtype=np.float64)*self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay,)
                )
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length,)
                )
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length,)
                )
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length*2 + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length*2 + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length*2,)
                )
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length*2, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length*2, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length*2,)
                )

            elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length,)
                )

            elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3, dtype=np.float64) * self.a,
                    high=np.ones(3, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3,)
                )
            else:
                raise Exception('Not Implemented')
        else:
            if self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + 1,)
                )
            elif self.time_dependency and self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length + 1,)
                )
            elif self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length + 1,)
                )
            elif self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length * 2 + self.max_delay + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length * 2 + self.max_delay + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length * 2 + 1,)
                )
            elif not self.time_dependency and self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length * 2 + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length * 2 + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length * 2 + 1,)
                )

            elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + self.prev_length + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.prev_length + 1,)
                )

            elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                self.observation_space = gym.spaces.Box(
                    low=np.ones(3 + 1, dtype=np.float64) * self.a,
                    high=np.ones(3 + 1, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + 1,)
                )
            else:
                raise Exception('Not Implemented')

        self.state = {}

        # Error catching
        assert isinstance(self.num_periods, int)

        # Check maximum possible order is less than inventory capacity for each node
        for i in range(len(self.order_max) - 1):
            for p in range(self.num_products):
                if self.order_max[i][p] > self.inv_max[i + 1][p]:
                    break
                    raise Exception('Maximum order cannot exceed maximum inventory of upstream node')



        # Maximum order of first node cannot exceed its own inventory
        assert (self.order_max[0][:] <= self.inv_max[0][:]).all()

        self.reset()

        # Grouping agents
        self.agent_types = {}
        self.view_space = {} 
        self.feature_space = self.feature_space_fn()
        
        #create agent types dictionary 
        self.setup_agents()

        self.group_handles = []
        for item in range(self.num_products):
            self.group_handles.append(str(item))
        self.group_handles_dict = {i: [] for i in self.group_handles}
        self.add_agent_to_group()

    
    def setup_agents(self):
        for idx, node_name in enumerate(self.node_names):
            self.agent_types[node_name] = idx 

        

    def add_agent_to_group(self):
        for i in self.group_handles:
            for item in self.node_names:
                if item[-1] == i:
                    self.group_handles_dict[i].append(item)
    

    def get_num(self, i, group_handles_dict):
        if i in group_handles_dict.keys():
            return len(self.group_handles_dict[i])
        else:
            print(f"Group handle {i} not recognized.")
            return 0

    def get_action_space(self, handle):
        for i in self.group_handles:
            return self.action_space.shape
        #this funion is useful in case the action space is different for different types of agents 

    def get_agent_ids(self, handle):
        return self.group_handles_dict[handle]

    def feature_space_fn(self):
        """
        Return the Box class (class from gym), Box.shape -> return shape.

        Returns
        -------
        feature_space: Box
            The Box class (class from gym).
        """
        fea_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(3,),
            dtype=np.uint8,
        )
        return fea_space


    def reset(self, customer_demand=None, noisy_delay=True, noisy_delay_threshold=0.1):
        """
        Create and initialize all variables.
        Nomenclature:
            inv = On hand inventory at the start of each period at each node (except last one).
            order_u = Pipeline inventory at the start of each period at each node (except last one).
            order_r = Replenishment order placed at each period at each node (except last one).
            demand = demand at each node
            ship = Sales performed at each period at each node.
            backlog = Backlog at each period at each node.
            profit = Total profit at each node.
        """

        periods = self.num_periods
        num_nodes = self.num_nodes
        num_products = self.num_products   

        self.prev_mean_action = [dict() for _ in range(self.num_types)]
        for agent in self.node_names:
            for action_type in range(self.num_types):
                self.prev_mean_action[action_type][agent] = np.zeros(self.action_space.shape[0])

        init_m_act = self.prev_mean_action

        if noisy_delay:
            self.noisy_delay = noisy_delay
            self.noisy_delay_threshold = noisy_delay_threshold

        if customer_demand is not None:
            self.customer_demand = customer_demand
        else:
            # Custom customer demand
            if self.demand_dist == "custom":
                self.customer_demand = self.config.get("customer_demand",
                                                    np.ones((len(self.retailers), self.num_periods, self.num_products),
                                                            dtype=np.int16) * 5)
                min_demand = 0
                max_demand = 30  # Adjust the range based on your requirements
                self.customer_demand = np.random.randint(min_demand, max_demand + 1, 
                                             size=(len(self.retailers), self.num_periods, self.num_products))
                amplitude = 10  # Amplitude of the sine wave
                offset = 15  # Offset to ensure \(\lambda\) is always positive
                periods_a = np.arange(self.num_periods)

                # Generate a sine wave for \(\lambda\)
                lambda_wave = amplitude * np.sin(2 * np.pi * periods_a / self.num_periods) + offset
                
                # Ensure \(\lambda\) is always positive
                lambda_wave = np.clip(lambda_wave, 0, None)
                    # Generate Poisson-distributed demand for each period, retailer, and product
                for t in periods_a:
                    for retailer in range(len(self.retailers)):
                        for product in range(self.num_products):
                            self.customer_demand[retailer, t, product] = np.random.poisson(lambda_wave[t])

            # Poisson distribution
            elif self.demand_dist == "poisson":
                self.mu = self.config.get("mu", 5)
                self.dist = poisson
                self.dist_param = {'mu': self.mu}
                self.customer_demand = self.dist.rvs(size=(len(self.retailers), self.num_periods, self.num_products), **self.dist_param)
            # Uniform distribution
            elif self.demand_dist == "uniform":
                lower_upper = self.config.get("lower_upper", (1, 5))
                lower = lower_upper[0]
                upper = lower_upper[1]
                self.dist = randint
                self.dist_param = {'low': lower, 'high': upper}
                if lower >= upper:
                    raise Exception('Lower bound cannot be larger than upper bound')
                self.customer_demand = self.dist.rvs(size=(len(self.retailers), self.num_periods, self.num_products), **self.dist_param)
            else:
                raise Exception('Unrecognised, Distribution Not Implemented')

            if self.noisy_demand:
                for k in range(len(self.retailers)):
                    for j in range(self.num_periods):
                        double_demand = np.random.uniform(0, 1)
                        zero_demand = np.random.uniform(0, 1)
                        if double_demand <= self.noisy_demand_threshold:
                            self.customer_demand[k, j] = 2 * self.customer_demand[k, j]
                        if zero_demand <= self.noisy_demand_threshold:
                            self.customer_demand[k, j] = 0

        # Assign customer demand to each product in each retailer
        self.retailer_demand = dict()
        for i in range(self.customer_demand.shape[0]):
            self.retailer_demand[self.retailers[i]] = dict()
            for p in range(self.num_products):
                self.retailer_demand[self.retailers[i]][p] = self.customer_demand[i,:,p]

        # simulation result lists
        self.inv = np.zeros([periods + 1, num_nodes, num_products])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_nodes, num_products])  # replenishment order (last node places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num_nodes, num_products])  # Unfulfilled order
        self.ship = np.zeros([periods, num_nodes, num_products])  # units sold
        self.acquisition = np.zeros([periods, num_nodes, num_products])
        self.backlog = np.zeros([periods + 1, num_nodes, num_products])  # backlog
        self.demand = np.zeros([periods + 1, num_nodes, num_products])
        if self.time_dependency:
            self.time_dependent_state = np.zeros([periods, num_nodes, num_products, self.max_delay])


        # Initialise list of dicts tracking products shipped from one node to another
        self.ship_to_list = []
        for i in range(self.num_periods):
            # Shipping dict
            ship_to = dict()
            for node in self.non_retailers:
                ship_to[node] = dict()
                for d_node in self.connections[node]:
                    ship_to[node][d_node] = dict()
                    for product in range(self.num_products):
                        ship_to[node][d_node][product] = 0

            self.ship_to_list.append(ship_to)

        # Initialise dict tracking backlog of products for each node
        self.backlog_to = dict()
        for i in range(self.num_nodes):
            if len(self.connections[i]) > 1:
                self.backlog_to[i] = dict()
                for node in self.connections[i]:
                    self.backlog_to[i][node] = dict()
                    for product in range(self.num_products):
                        self.backlog_to[i][node][product] = 0

        # Initialisation
        self.period = 0  # initialise time
        for node in self.retailers:
            for product in range(self.num_products):
                self.demand[self.period, node, product] = self.retailer_demand[node][product][self.period]
        self.inv[self.period, :, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()
        #TODO: this has changed, m_act has been hardcoded to 0 for reset for now 
        init_set = {'obs': self.state, 'm_act': init_m_act}
        return init_set

    def _update_state(self):
        # Dictionary containing observation of each agent
        obs = {}

        t = self.period
        m = self.num_nodes
        for i in range(m):
            for p in range(self.num_products):
            # Each agent observes five things at every time-step
            # Their inventory, backlog, demand received, acquired inventory from upstream node
            # and inventory sent to downstream node which forms an observation/state vecto
                agent = 'node_' + str(i) + str(p)
                # Initialise state vector
                if not self.share_network:
                    if self.time_dependency and not self.prev_actions and not self.prev_demand:
                        obs_vector = np.zeros(3 + self.max_delay)
                    elif self.time_dependency and self.prev_actions and not self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length + self.max_delay)
                    elif self.time_dependency and not self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length + self.max_delay)
                    elif self.time_dependency and self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length*2 + self.max_delay)
                    elif not self.time_dependency and self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length*2)
                    elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length)
                    elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                        obs_vector = np.zeros(3)
                else:
                    if self.time_dependency and not self.prev_actions and not self.prev_demand:
                        obs_vector = np.zeros(3 + self.max_delay + 1)
                    elif self.time_dependency and self.prev_actions and not self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length + self.max_delay + 1)
                    elif self.time_dependency and not self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length + self.max_delay + 1)
                    elif self.time_dependency and self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length*2 + self.max_delay + 1)
                    elif not self.time_dependency and self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length*2 + 1)
                    elif not self.time_dependency and not self.prev_actions and self.prev_demand:
                        obs_vector = np.zeros(3 + self.prev_length + 1)
                    elif not self.time_dependency and not self.prev_actions and not self.prev_demand:
                        obs_vector = np.zeros(3 + 1)

                if self.prev_demand:
                    demand_history = np.zeros(self.prev_length)
                    for j in range(self.prev_length):
                        if j < t:
                            demand_history[j] = self.demand[t - 1 - j, i, p] #period, node, product
                    demand_history = self.rescale(demand_history, np.zeros(self.prev_length),
                                                np.ones(self.prev_length)*self.demand_max[i][p],
                                                self.a, self.b)

                if self.prev_actions:
                    order_history = np.zeros(self.prev_length)
                    for j in range(self.prev_length):
                        if j < t:
                            order_history[j] = self.order_r[t - 1 - j, i, p] #period, node, product
                    order_history = self.rescale(order_history, np.zeros(self.prev_length),
                                                np.ones(self.prev_length)*self.order_max[i][p],
                                                self.a, self.b)

                if self.time_dependency:
                    delay_states = np.zeros(self.max_delay)
                    if t >= 1:
                        delay_states = np.minimum(self.time_dependent_state[t - 1, i, p, :], np.ones(self.max_delay)*self.inv_max[i]*2)
                    delay_states = self.rescale(delay_states, np.zeros(self.max_delay),
                                                                        np.ones(self.max_delay)*self.inv_max[i][p]*2,  # <<<<<<
                                                                        self.a, self.b)

                obs_vector[0] = self.rescale(self.inv[t, i, p], 0, self.inv_max[i][p], self.a, self.b)
                obs_vector[1] = self.rescale(self.backlog[t, i, p], 0, self.demand_max[i][p], self.a, self.b)
                obs_vector[2] = self.rescale(self.order_u[t, i, p], 0, self.order_max[i][p], self.a, self.b)
                if self.time_dependency and not self.prev_actions and not self.prev_demand:
                    obs_vector[3:3+self.max_delay] = delay_states
                elif self.time_dependency and self.prev_actions and not self.prev_demand:
                    obs_vector[3:3+self.prev_length] = order_history
                    obs_vector[3+self.prev_length:3+self.prev_length+self.max_delay] = delay_states
                elif self.time_dependency and not self.prev_actions and self.prev_demand:
                    obs_vector[3:3+self.prev_length] = demand_history
                    obs_vector[3+self.prev_length:3+self.prev_length+self.max_delay] = delay_states
                elif self.time_dependency and self.prev_actions and self.prev_demand:
                    obs_vector[3:3+self.prev_length] = demand_history
                    obs_vector[3+self.prev_length:3+self.prev_length*2] = order_history
                    obs_vector[3+self.prev_length*2:3+self.prev_length*2+self.max_delay] = delay_states
                elif not self.time_dependency and self.prev_actions and not self.prev_demand:
                    obs_vector[3:3 + self.prev_length] = demand_history
                elif not self.time_dependency and self.prev_actions and self.prev_demand:
                    obs_vector[3:3 + self.prev_length] = demand_history
                    obs_vector[3 + self.prev_length:3 + self.prev_length * 2] = order_history

                if self.share_network:
                    obs_vector[len(obs_vector) - 1] = self.rescale(i, 0, self.num_nodes, self.a, self.b)

                obs[agent] = obs_vector

        self.state = obs.copy()

        return self.state

    def step(self, action_dict):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_nodes
        
        # Get the mean action for the one hop neighbourhood for each agent 
        self.action_mean = [dict() for _ in range(self.num_types)]

        self.m_act = {}
        for node_name in self.node_names:
            one_hop_actions = []
            one_hop_connections = find_connections(node_name, self.adjacency, self.node_names)
            for i in one_hop_connections:
                one_hop_actions.append(action_dict[i])
            self.m_act[node_name] = np.mean(one_hop_actions)
        
        for node_name in self.node_names:
            for action_type in range(self.num_types):
                self.prev_mean_action[action_type][node_name] = self.m_act[node_name]
                self.action_mean[action_type][node_name] = self.m_act[node_name]

        


        # Get replenishment order at each node
        for i in range(self.num_nodes):
            for p in range(self.num_products):
                node_name = "node_" + str(i) + str(p)
                self.order_r[t, i, p] = self.rev_scale(action_dict[node_name], 0, self.order_max[i][p], self.a, self.b)
                self.order_r[t, i, p] = np.round(self.order_r[t, i, p], 0).astype(int)

        self.order_r[t, :, :] = np.minimum(np.maximum(self.order_r[t, :, :], np.zeros((self.num_nodes, self.num_products))), self.order_max)

        # Demand of goods at each stage
        # Demand at last (retailer stages) is customer demand
        for node in self.retailers:
            for product in range(self.num_products):
                self.demand[t, node, product] = np.minimum(self.retailer_demand[node][product][t], self.inv_max[node][product])  # min for re-scaling
        # Demand at other stages is the replenishment order of the downstream stage
        for i in range(self.num_nodes):
            for p in range(self.num_products):
                if i not in self.retailers:
                    for j in range(i, len(self.network[i])):
                        if self.network[i][j] == 1:
                            self.demand[t, i, p] += self.order_r[t, j, p]

        # Update acquisition, i.e. goods received from previous node
        self.update_acquisition()

        # Amount shipped by each node to downstream node at each time-step. This is backlog from previous time-steps
        # And demand from current time-step, This cannot be more than the current inventory at each node
        self.ship[t, :, :] = np.minimum(self.backlog[t, :, :] + self.demand[t, :, :], self.inv[t, :, :] + self.acquisition[t, :, :])

        # Get amount shipped to downstream nodes
        for i in self.non_retailers:
            for p in range(self.num_products):
                # If shipping to only one downstream node, the total amount shipped is equivalent to amount shipped to
                # downstream node
                if self.num_downstream[i] == 1:
                    self.ship_to_list[t][i][self.connections[i][0]][p] = self.ship[t, i, p]
                # If node has more than one downstream nodes, then the amount shipped needs to be split appropriately
                elif self.num_downstream[i] > 1:
                    # Extract the total amount shipped in this period
                    ship_amount = self.ship[t, i, p]
                    # If shipment equal to or more than demand, send ordered amount to each downstream node
                    if self.ship[t, i, p] >= self.demand[t, i, p]:
                        # If there is backlog, fulfill it first then fulfill demand
                        if self.backlog[t, i, p] > 0:
                            # Fulfill backlog first
                            while_counter = 0  # to exit infinite loops if error
                            # Keep distributing shipment across downstream nodes until there is no backlog or no goods left
                            sum_backlog = 0 
                            for key in self.backlog_to[i]:
                                sum_backlog += self.backlog_to[i][key][p]
                            while sum_backlog > 0 and ship_amount > 0:
                                # Keep distributing shipped goods to downstream nodes
                                for node in self.connections[i]:
                                    # If there is a backlog towards a downstream node ship a unit of product to that node
                                    if self.backlog_to[i][node][p] > 0:
                                        self.ship_to_list[t][i][node][p] += 1  # increase amount shipped to node
                                        self.backlog_to[i][node][p] -= 1  # decrease its corresponding backlog
                                        ship_amount -= 1  # reduce amount of shipped goods left

                                #recalculate sum backlog for while condition check
                                sum_backlog = 0 
                                for key in self.backlog_to[i]:
                                    sum_backlog += self.backlog_to[i][key][p]
                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i][p] * 2:
                                    raise Exception("Infinite Loop 1")

                            # If there is still left-over shipped goods fulfill current demand if any
                            if ship_amount > 0 and self.demand[t, i, p] > 0:
                                # Create a dict of downstream nodes' demand/orders
                                outstanding_order = dict()
                                for node in self.connections[i]:
                                    if node not in outstanding_order:
                                        outstanding_order[node] = dict()
                                    outstanding_order[node][p] = self.order_r[t, node, p]

                                while_counter = 0
                                # Keep distributing shipment across downstream nodes until there is no backlog or no
                                # outstanding orders left
                                sum_outstanding_order = 0
                                for node in self.connections[i]:
                                    sum_outstanding_order += outstanding_order[node][p]
                                    
                                while ship_amount > 0 and sum_outstanding_order > 0:
                                    for node in self.connections[i]:
                                        if outstanding_order[node][p] > 0:
                                            self.ship_to_list[t][i][node][p] += 1  # increase amount shipped to node
                                            outstanding_order[node][p] -= 1  # decrease its corresponding outstanding order
                                            ship_amount -= 1  # reduce amount of shipped goods left

                                    #recalculate sum outstanding order for while condition check
                                    sum_outstanding_order = 0
                                    for node in self.connections[i]:
                                        sum_outstanding_order += outstanding_order[node][p]

                                    # Counter to escape while loop with error if infinite
                                    while_counter += 1
                                    if while_counter > self.demand_max[i][p]:
                                        raise Exception("Infinite Loop 2")

                                # Update backlog if some outstanding order unfulfilled
                                for node in self.connections[i]:
                                    self.backlog_to[i][node][p] += outstanding_order[node][p]

                        # If there is no backlog
                        else:
                            for node in self.connections[i]:
                                self.ship_to_list[t][i][node][p] += self.order_r[t, node, p]
                                ship_amount = ship_amount - self.order_r[t, node, p]
                            if ship_amount > 0:
                                print("WTF")

                    # If shipment is insufficient to meet downstream demand
                    elif self.ship[t, i, p] < self.demand[t, i, p]:
                        while_counter = 0
                        # Distribute amount shipped to downstream nodes
                        if self.backlog[t, i, p] > 0:
                            # Fulfill backlog first
                            while_counter = 0  # to exit infinite loops if error
                            # Keep distributing shipment across downstream nodes until there is no backlog or no goods left
                            sum_backlog = 0 
                            for key in self.backlog_to[i]:
                                sum_backlog += self.backlog_to[i][key][p]
                            while sum_backlog > 0 and ship_amount > 0:
                                # Keep distributing shipped goods to downstream nodes
                                for node in self.connections[i]:
                                    # If there is a backlog towards a downstream node ship a unit of product to that node
                                    if self.backlog_to[i][node][p] > 0:
                                        self.ship_to_list[t][i][node][p] += 1  # increase amount shipped to node
                                        self.backlog_to[i][node][p] -= 1  # decrease its corresponding backlog
                                        ship_amount -= 1  # reduce amount of shipped goods left

                                #recalculate sum backlog for while condition check
                                sum_backlog = 0
                                for key in self.backlog_to[i]:
                                    sum_backlog += self.backlog_to[i][key][p]

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i][p]:
                                    raise Exception("Infinite Loop 3")

                        else:
                            # Keep distributing shipped goods to downstream nodes until no goods left
                            while ship_amount > 0:
                                for node in self.connections[i]:
                                    # If amount being shipped less than amount ordered
                                    if self.ship_to_list[t][i][node][p] < self.order_r[t, node, p] + self.backlog_to[i][node][p]:
                                        self.ship_to_list[t][i][node][p] += 1  # increase amount shipped to node
                                        ship_amount -= 1  # reduce amount of shipped goods left

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i][p]:
                                    raise Exception("Infinite Loop 4")

                        # Log unfulfilled order amount as backlog
                        for node in self.connections[i]:
                            self.backlog_to[i][node][p] += self.order_r[t, node, p] - self.ship_to_list[t][i][node][p]

        # Update backlog demand increases backlog while fulfilling demand reduces it
        self.backlog[t + 1, :, :] = self.backlog[t, :, :] + self.demand[t, :, :] - self.ship[t, :, :]
        # Capping backlog to allow re-scaling
        self.backlog[t + 1, :, :] = np.minimum(self.backlog[t + 1, :, :], self.demand_max)

        # Update time-dependent states
        if self.time_dependency:
            self.time_dependent_acquisition()

        # Update unfulfilled orders/ pipeline inventory
        self.order_u[t + 1, :, :] = np.minimum(
            np.maximum(
                self.order_u[t, :, :] + self.order_r[t, :, :] - self.acquisition[t, :, :],
                np.zeros((self.num_nodes, self.num_products))),
            self.inv_max)


        # Update inventory
        self.inv[t + 1, :, :] = np.minimum(
            np.maximum(
                self.inv[t, :, :] + self.acquisition[t, :, :] - self.ship[t, :, :],
                np.zeros((self.num_nodes, self.num_products))),
            self.inv_max)

        # Calculate rewards
        rewards, profit = self.get_rewards()

        # Update period
        self.period += 1

        # Update state
        upd_state = self._update_state()

        # determine if simulation should terminate
        done = {
            "__all__": self.period >= self.num_periods,
        }

        info = {}
        for i in range(self.num_nodes):
            for p in range(self.num_products):
                node = 'node_' + str(i) + str(p)
                meta_info = dict()
                meta_info['period'] = self.period
                meta_info['demand'] = self.demand[t, i, p]
                meta_info['ship'] = self.ship[t, i, p]
                meta_info['acquisition'] = self.acquisition[t, i, p]
                meta_info['actual order'] = self.order_r[t, i, p]
                meta_info['profit'] = profit[i]
                info[node] = meta_info

        #TODO: adding feature space for future use. currently a dummy 
        fea = self.state.copy()
        
        """for key, value in action_dict.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Value for key '{key, value}' is not a NumPy array. Value type: {type(value)}")
"""
        return self.state, action_dict, rewards, self.action_mean, upd_state, fea 

    def get_rewards(self):
        rewards = {}
        profit = np.zeros(self.num_nodes)
        m = self.num_nodes
        t = self.period
        reward_sum = 0
        for i in range(self.num_nodes):
            for p in range(self.num_products):
                agent = 'node_' + str(i) + str(p)
                reward = self.node_price[i][p] * self.ship[t, i, p] \
                    - self.node_cost[i][p] * self.order_r[t, i, p] \
                    - self.stock_cost[i][p] * np.abs(self.inv[t + 1, i, p] - self.inv_target[i][p]) \
                    - self.backlog_cost[i][p] * self.backlog[t + 1, i, p]

                reward_sum += reward
                profit[i] = reward
                if self.independent:
                    rewards[agent] = reward

        if not self.independent:
            for i in range(self.num_nodes):
                for p in range(self.num_products):
                    agent = 'node_' + str(i) + str(p)
                    rewards[agent] = reward_sum/(self.num_nodes*self.num_products)

        return rewards, profit

    def update_acquisition(self):
        """
        Get acquisition at each node
        :return: None
        """
        m = self.num_nodes
        t = self.period

        # Acquisition at node 0 is unique since delay is manufacturing delay instead of shipment delay
        for p in range(self.num_products):
            if t - int(self.delay[0,p]) >= 0:
                extra_delay = False
                if self.noisy_delay:
                    delay_percent = np.random.uniform(0, 1)
                    if delay_percent <= self.noisy_delay_threshold:
                        extra_delay = True

                if self.acquisition[t, 0, :].shape != self.order_r[t - self.delay[0,p], 0, :].shape:
                    self.acquisition[t, 0, :] += np.squeeze(self.order_r[t - self.delay[0,p], 0, :]) 
                else:
                    self.acquisition[t, 0, :] += self.order_r[t - self.delay[0,p], 0, :]

                if extra_delay and t < self.num_periods - 1:
                    self.acquisition[t + 1, 0, :] += self.acquisition[t, 0, :]
                    self.acquisition[t, 0, :] = 0
            else:
                self.acquisition[t, 0, :] = self.acquisition[t, 0, :]



        # Acquisition at subsequent stage is the delayed shipment of the upstream stage
        for i in range(1, m):
            for p in range(self.num_products):
                if t - self.delay[i,p] >= 0:
                    extra_delay = False
                    if self.noisy_delay:
                        delay_percent = np.random.uniform(0, 1)
                        if delay_percent <= self.noisy_delay_threshold:
                            extra_delay = True

                    self.acquisition[t, i, p] += \
                    self.ship_to_list[int(t - self.delay[i,p])][self.upstream_node[i]][i][p]
                    if extra_delay and t < self.num_periods - 1:
                        self.acquisition[t + 1, i, p] += self.acquisition[t, i, p]
                        self.acquisition[t, i, p] = 0

                else:
                    self.acquisition[t, i, p] = self.acquisition[t, i, p]

    def time_dependent_acquisition(self):
        """
        Get time-dependent states
        :return: None
        """
        m = self.num_nodes
        t = self.period

        # Shift delay down with every time-step
        if self.max_delay > 1 and t >= 1:
            self.time_dependent_state[t, :, :, 0:self.max_delay - 1] = self.time_dependent_state[t - 1, :, :
                                                                    1:self.max_delay]

        # Delayed states of first node
        self.time_dependent_state[t, 0, :, self.delay[0] - 1] = self.order_r[t, 0, :]
        # Delayed states of rest of n:
        for i in range(1, m):
            for p in range(self.num_products):
                self.time_dependent_state[t, i, p, self.delay[i] - 1] = \
                    self.ship_to_list[t][self.upstream_node[i]][i][p]

    def rescale(self, val, min_val, max_val, A=-1, B=1):
        if isinstance(val, np.ndarray):
            a = np.ones(np.size(val)) * A
            b = np.ones(np.size(val)) * B
        else:
            a = A
            b = B

        val_scaled = a + (((val - min_val) * (b - a)) / (max_val - min_val))

        return val_scaled

    def rev_scale(self, val_scaled, min_val, max_val, A=-1, B=1):
        if isinstance(val_scaled, np.ndarray):
            a = np.ones(np.size(val_scaled)) * A
            b = np.ones(np.size(val_scaled)) * B
        else:
            a = A
            b = B

        val = (((val_scaled - a) * (max_val - min_val)) / (b - a)) + min_val

        return val
    

def test():
    config = {}
    env = MultiAgentInvManagementDiv(config)
    ini_st = env.reset()
    done = False
    for _ in range(365):
        action_dict = {}
        for i in range(env.num_nodes):
            for p in range(env.num_products):
                action_dict['node_' + str(i) + str(p)] = np.random.uniform(env.a, env.b)
        state, action_dict, rewards, action_mean, upd_state, fea  = env.step(action_dict)

test()
print('Done')


