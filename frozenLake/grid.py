import random
import numpy as np

def draw_grid(l = 4, b = 4):
    grid = []
    char_list = ['-', '*']
    prob_of_danger = 0.2
    start = 'S'
    goal = 'G'
    for i in range(l):
        row = []
        for j in range(b):
            row.append(char_list[random.random() < prob_of_danger])
        prob_of_danger = max(prob_of_danger-0.05, 0)
        grid.append(row)
    grid[0][0] = start
    grid[l-1][b-1] = goal

    return np.asarray(grid)
