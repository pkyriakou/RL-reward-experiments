import random
import numpy as np
import copy
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

num_action = 3
state_size = 10 * 10


class task3():

    def __init__(self, r_seed=0, render=False):
        random.seed(r_seed)
        self.UI = render
        self.size = 10
        self.p = self.rain(self.size)
        self.position = int(self.size / 2)

        img = copy.deepcopy(self.p.screen)
        img[0][self.position] = 0.7
        img.reverse()
        self.state = np.array(img) * 255
        self.steps = 0
        self.avg_reward = 0

    def step(self, action):
        # print(self.state)
        self.steps += 1
        if action == 0:
            reward = self.go_left()
        elif action == 1:
            reward = self.nothing()
        elif action == 2:
            reward = self.go_right()

        img = copy.deepcopy(self.p.screen)
        img[0][self.position] = 0.7
        img.reverse()
        img = np.array(img) * 255
        self.state = img
        self.avg_reward += (1 / self.steps) * (reward - self.avg_reward)
        if self.UI:
            self.render()
        return img, reward

    def go_left(self):
        if self.position > 0:
            if self.p.screen[1][self.position - 1] == 1:
                reward = 1
            else:
                reward = 0

            self.p.forward()
            self.position -= 1

        else:
            reward = self.nothing()
        return reward

    def go_right(self):
        if self.position < (self.size - 1):
            if self.p.screen[1][self.position + 1] == 1:
                reward = 1
            else:
                reward = 0

            self.p.forward()
            self.position += 1

        else:
            reward = self.nothing()

        return reward

    def nothing(self):
        if self.p.screen[1][self.position] == 1:
            reward = 1
        else:
            reward = 0
        self.p.forward()
        return reward

    def render(self):
        cvals = [-1, 1]
        colors = ["black", "white"]
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        a = self.zoom()
        plt.clf()
        plt.imshow(a, cmap=cmap)

        prop = fm.FontProperties(fname='PressStart2P.ttf')
        plt.title('average reward = %.3f' % self.avg_reward, fontproperties=prop, size=10, pad=7)
        plt.axis('off')
        plt.pause(0.1)

    def zoom(self):
        img = self.state
        x = 6
        IMG = []
        for i in img:
            for k in range(x):
                b = []
                for j in i:
                    b.extend([j] * x)
                IMG.append(b)
        IMG = np.array(IMG)
        return IMG

    class rain():
        def __init__(self, size):
            self.size = size
            self.screen = self.init()

        def init(self):
            rain = [self.get_row() for i in range(self.size - 1)] + [[0] * self.size]
            return rain

        def forward(self):
            self.screen = self.screen[1:]
            row = self.get_row()
            self.screen.append(row)

        def get_row(self):
            row = [0] * self.size
            if random.random() > 0.5:
                row[random.randint(0, self.size - 1)] = 1
            return row
