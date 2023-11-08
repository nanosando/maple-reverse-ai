import os
import Arena
from MCTS import MCTS
from reverse.ReverseGame import ReverseGame
from reverse.ReversePlayers import *
from reverse.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

import matplotlib.pyplot as plt

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

num_play = 100
# display = ReverseGame.display
display = None
verbose = False
if display: verbose = True
g = ReverseGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyReversePlayer(g).play

x = []
y_random = []
y_greedy = []

prefix = './temp/'
for i in range(100):
    filename = f'checkpoint_{i}.pth'
    if not os.path.isfile(prefix + filename):
        continue

    print(f'Evaluating {filename}...')

    # nnet player
    n1 = NNet(g)
    n1.load_checkpoint(prefix, filename)

    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    # set to random player
    print('AI versus random player')
    player2 = rp
    arena = Arena.Arena(n1p, player2, g, display=display)
    oneWon, twoWon, draws = arena.playGames(num_play, verbose=verbose)
    random_rate = 100 * oneWon / (oneWon + twoWon + draws)
    print(f'Winning Rate: {random_rate} % ({oneWon}, {twoWon}, {draws})\n')

    # set to greedy player
    print('AI versus greedy player')
    player2 = gp
    arena = Arena.Arena(n1p, player2, g, display=display)
    oneWon, twoWon, draws =  arena.playGames(num_play, verbose=verbose)
    greedy_rate = 100 * oneWon / (oneWon + twoWon + draws)
    print(f'Winning Rate: {greedy_rate} % ({oneWon}, {twoWon}, {draws})\n')

    x.append(i)
    y_random.append(random_rate)
    y_greedy.append(greedy_rate)

plt.figure(figsize=(10,10))
plt.plot(x, y_random, label='Random Player')
plt.plot(x, y_greedy, label='Greedy Player')
plt.xlabel("Iteration")
plt.ylabel("Winning rate (%)")
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.legend()
plt.savefig('./fig/winning_rate.png')
plt.clf()