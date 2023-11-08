import Arena
from MCTS import MCTS
from reverse.ReverseGame import ReverseGame
from reverse.ReversePlayers import *
from reverse.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

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

# nnet player
n1 = NNet(g)
n1.load_checkpoint('./temp/', 'best.pth')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# set to random player
print('AI versus random player')
player2 = rp
arena = Arena.Arena(n1p, player2, g, display=display)
oneWon, twoWon, draws = arena.playGames(num_play, verbose=verbose)
print(f'Winning Rate: {100 * oneWon / (oneWon + twoWon + draws)} % ({oneWon}, {twoWon}, {draws})\n')

# set to greedy player
print('AI versus greedy player')
player2 = gp
arena = Arena.Arena(n1p, player2, g, display=display)
oneWon, twoWon, draws =  arena.playGames(num_play, verbose=verbose)
print(f'Winning Rate: {100 * oneWon / (oneWon + twoWon + draws)} % ({oneWon}, {twoWon}, {draws})\n')