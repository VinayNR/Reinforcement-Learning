import random
import math
import numpy as np
import grid
import argparse

actions = ['R', 'L', 'U', 'D']

def choose_action(q_table, current_state, explore_rate, breadth):
    if random.random() < explore_rate:
        return random.choice(actions)
    return actions[q_table[current_state[0]*breadth + current_state[1]].argmax()]

def perform_action(current_state, action, length, breadth):
    x = current_state[0]
    y = current_state[1]
    if action == 'L':
        if current_state[1] == 0:
            return (x,y)
        else:
            return (x,y-1)

    elif action == 'R':
        if current_state[1] == breadth-1:
            return (x,y)
        else:
            return (x,y+1)

    elif action == 'U':
        if current_state[0] == 0:
            return (x,y)
        else:
            return (x-1,y)

    else:
        if current_state[0] == length-1:
            return (x,y)
        else:
            return (x+1,y)

def get_reward(board, state):
    if board[state[0]][state[1]] == 'G':
        return 10
    elif board[state[0]][state[1]] == '-':
        return 1
    else:
        return -10

def main(args):

    length = args.l
    breadth = args.b

    learning_rate = args.lr
    discount_rate = args.dr
    min_explore_rate = args.p
    max_explore_rate = args.q
    decay_factor = args.d

    num_episodes = args.ep
    num_iterations = args.iter

    board = grid.draw_grid(length, breadth)
    # board = [['S', '-', '*', '*', '-'], ['*', '-', '*', '*', '-'], ['-', '-', '*', '-', '-'], ['-', '*', '-', '-', '-'], ['-', '-', '-', '*', 'G']]
    print("Board -:\n\n", board)
    q_table = np.zeros((length*breadth, len(actions)))

    explore_rate = max_explore_rate
    i = 0
    while i < num_episodes:
        j = 0
        completed = False
        x = y = 0    # Start State
        while j < num_iterations and completed == False:
            current_state = (x,y)
            action = choose_action(q_table, current_state, explore_rate, breadth)
            next_state = perform_action(current_state, action, length, breadth)
            if board[next_state[0]][next_state[1]] == 'G':
                completed = True
            reward = get_reward(board, next_state)
            x_new, y_new = next_state
            q_table[x*breadth + y][actions.index(action)] = q_table[x*breadth + y][actions.index(action)] + learning_rate*(reward + discount_rate*q_table[x_new*breadth + y_new].max() - q_table[x*breadth + y][actions.index(action)])
            x,y = x_new, y_new
            explore_rate = min_explore_rate + (max_explore_rate - min_explore_rate)*np.exp(-decay_factor*(i*num_iterations + j))
            j = j + 1

        i = i + 1
        print("Episode {} : Iterations : {}".format(i,j))

    print("\nFinal Q-Table -:\n\n", q_table)

def parse_arguments():
    '''
    Parses the command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Frozen Lake.")

    parser.add_argument('--l', type=int, default=4,
                        help='Length of grid')

    parser.add_argument('--b', type=int, default=4,
                        help='Breadth of grid')

    parser.add_argument('--lr', type=float, default=0.5,
                        help='Learning Rate of Q-Learning')

    parser.add_argument('--dr', type=float, default=0.7,
                        help='Discount Factor of Q-Learning')

    parser.add_argument('--p', type=float, default=0.1,
                        help='Minimum explore rate')

    parser.add_argument('--q', type=float, default=1.0,
                        help='Maximum explore rate')

    parser.add_argument('--d', type=float, default=0.01,
                        help='Decay factor of explore rate')

    parser.add_argument('--ep', type=int, default=1000,
                        help='Number of episodes')

    parser.add_argument('--iter', type=int, default=1000,
                        help='Number of iterations per episode')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
