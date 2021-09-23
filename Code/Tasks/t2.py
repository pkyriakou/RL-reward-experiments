import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

state_size = 6
num_action = 6

PROD_TIME_PARAMS = [(8, 1 / 8), (8, 2 / 8), (8, 3 / 8), (8, 4 / 8), (8, 5 / 8)]
REWARDS = [9, 7, 16, 20, 25]
BUFFER_SIZES = [30, 20, 15, 15, 10]
DEMAND_ARRIVALS = [6, 9, 21, 26, 30]

MAINT_COST = 500
REPAIR_COST = 5000
MAINT_TIME_MIN = 20
MAINT_TIME_MAX = 40
REPAIR_TIME_SHAPE = 2
REPAIR_TIME_SCALE = 1 / 0.04
TIME_BET_FAILURE_SHAPE = 6
TIME_BET_FAILURE_SCALE = 1 / 0.02


class task2:
    def __init__(self, r_seed=0, render=False):
        np.random.seed(r_seed)
        self.UI = render
        # Event IDS:
        # 0-4: demand arrives for given product
        # 5: next product in production is complete
        # 6: failure
        # 7: repair over
        self.events = [None] * 8

        # 0-4: production of product
        self.current_production = None

        self.t = 0
        self.last_work = 0
        self.schedule_next_failure()
        self.products = [product(PROD_TIME_PARAMS[i], DEMAND_ARRIVALS[i], BUFFER_SIZES[i], REWARDS[i]) for i in
                         range(5)]
        for i in range(5):
            self.schedule_demand(i)
        self.state = self.get_state()
        self.avg_reward = 0
        self.steps = 0
        # print(self.events)

    def step(self, action):
        self.steps += 1
        reward = 0
        action_needed = False
        # print(self.state)

        if action < 5:
            self.schedule_production(action)
        elif action == 5:
            self.maintenance()
            reward -= MAINT_COST

        # this rolls out the action till a next one is needed
        while action_needed == False:
            event_id = self.argmin(self.events)
            self.t = self.events[event_id]

            # demand arrived, resquedule next demand
            if event_id < 5:
                reward += self.products[event_id].demand_arrival()
                self.schedule_demand(event_id)

            if event_id == 5:
                production_finished = self.products[self.current_production].add_to_buffer()
                if production_finished:
                    action_needed = True
                    self.current_production = None
                else:
                    self.schedule_production(self.current_production)

            if event_id == 6:
                self.repair()
                reward -= REPAIR_COST

            if event_id == 7:
                self.events[7] = None
                self.schedule_next_failure()
                action_needed = True
                self.last_work = self.t

        self.state = self.get_state()
        self.avg_reward += (1 / self.steps) * (reward - self.avg_reward)
        if self.UI:
            self.render()

        return self.state, reward

    def get_state(self):
        buffers = [p.buffer for p in self.products]
        time_since_work = self.t - self.last_work if self.last_work != None else 0
        return buffers + [time_since_work]

    def repair(self):
        self.events[6] = None  # no next failure scheduled until repair is done
        self.events[5] = None  # no production until repair is done
        self.current_production = None
        self.events[7] = np.random.gamma(REPAIR_TIME_SHAPE, REPAIR_TIME_SCALE) + self.t

    def maintenance(self):
        self.events[6] = None  # no next failure scheduled until repair is done
        self.events[5] = None  # no production until repair is done
        self.current_production = None
        self.events[7] = np.random.uniform(MAINT_TIME_MIN, MAINT_TIME_MAX) + self.t

    def schedule_demand(self, product_nr):
        self.events[product_nr] = self.products[product_nr].next_demand() + self.t

    def schedule_production(self, product_nr):
        self.events[5] = self.products[product_nr].prod_time() + self.t
        self.current_production = product_nr

    def schedule_next_failure(self):
        self.events[6] = np.random.gamma(TIME_BET_FAILURE_SHAPE, TIME_BET_FAILURE_SCALE) + self.t

    def argmin(self, seq):
        index = None
        minimum = float("inf")
        for i in range(len(seq)):
            curr = seq[i]
            if curr != None and curr < minimum:
                index = i
                minimum = curr
        return index

    def render(self):

        cvals = [-1, 1]
        colors = ["black", "white"]
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        plt.clf()

        a = self.get_img()
        plt.imshow(a, cmap=cmap)

        prop = fm.FontProperties(fname='PressStart2P.ttf')
        plt.title('average reward = %.3f' % self.avg_reward, fontproperties=prop, size=10, pad=7)
        plt.text(1, 232, 'last fix: %0.f' % self.state[5], fontproperties=prop, size=10, color='white')
        plt.axis('off')
        plt.pause(0.1)

    def get_img(self):
        s = self.state
        width = 30
        spacing = 10
        rows = []

        SCALE = 7

        for i in range(spacing):
            rows.append(35 * SCALE * [0])

        for i in range(5):
            occupied = s[i] * SCALE
            free = (BUFFER_SIZES[i] * SCALE - occupied)
            blank = (35 * SCALE - free - occupied)

            row = [255.] * occupied + [120.] * free + [0.] * blank

            for i in range(width):
                rows.append(row)

            for i in range(spacing):
                rows.append(35 * SCALE * [0])

        for i in range(2):
            rows.append(35 * SCALE * [255])

        for i in range(30):
            rows.append(35 * SCALE * [0])

        return np.array(rows)


class product:
    def __init__(self, prod_time_param, demand_arrival_param, buffer_size, reward):
        self.buffer = 0
        self.buffer_size = buffer_size
        self.reward = reward
        self.prod_time_param = prod_time_param
        self.demand_arrival_param = demand_arrival_param
        self.produced_since_failure = 0

    def next_demand(self):
        return np.random.poisson(self.demand_arrival_param)

    def prod_time(self):
        shape, scale = self.prod_time_param
        return np.random.gamma(shape, scale)

    def add_to_buffer(self):
        if self.buffer < self.buffer_size:
            self.buffer += 1
        return self.buffer == self.buffer_size

    def demand_arrival(self):
        if self.buffer > 0:
            self.buffer -= 1
            return self.reward
        else:
            return 0
