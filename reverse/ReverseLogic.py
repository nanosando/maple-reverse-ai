'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

'''
Added 5 random holes for Maple reverse
'''

import random

class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        self.n_blank = 5
        
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Set up the initial 4 pieces.
        self.pieces[int(self.n/2)-1][int(self.n/2)] = 1
        self.pieces[int(self.n/2)][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)-1] = -1
        self.pieces[int(self.n/2)][int(self.n/2)] = -1

        # Make n_blank holes on the board
        for i in range(self.n_blank):
            while True:
                y = random.randrange(self.n)
                x = random.randrange(self.n)
                if self.is_blank_valid(y, x):
                    self.pieces[y][x] = 2
                    break

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[y][x]==color:
                    count += 1
                if self[y][x]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[y][x]==color:
                    newmoves = self.get_moves_for_square((y, x))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((y, x))
                    if newmoves and len(newmoves) > 0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (y, x) = square

        # determine the color of the piece.
        color = self[y][x]

        # skip empty and hole squares.
        if color == 0 or color == 2 or color == -2:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [flip for direction in self.__directions
                      for flip in self._get_flips(move, direction, color)]
        assert len(list(flips))>0
        for y, x in flips:
            #print(self[x][y],color)
            self[y][x] = color

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        y, x = origin
        color = self[y][x]
        flips = []

        for y, x in Board._increment_move(origin, direction, self.n):
            if self[y][x] == 0:
                if flips:
                    # print("Found", x,y)
                    return (y, x)
                else:
                    return None
            elif self[y][x] == color:
                return None
            elif self[y][x] == -color:
                # print("Flip",x,y)
                flips.append((y, x))
            elif self[y][x] == 2 or self[y][x] == -2:
                return None

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for y, x in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[y][x] == 0 or self[y][x] == 2 or self[y][x] == -2:
                return []
            if self[y][x] == -color:
                flips.append((y, x))
            elif self[y][x] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

    # Custom
    def is_blank_valid(self, y, x):
        if self.pieces[y][x] != 0:
            return False
        else:
            # Up
            if y >= 1 and (self.pieces[y - 1][x] == 2 or self.pieces[y - 1][x] == -2):
                return False

            # Down
            if y < self.n - 1 and (self.pieces[y + 1][x] == 2 or self.pieces[y + 1][x] == -2):
                return False

            # Left
            if x >= 1 and (self.pieces[y][x - 1] == 2 or self.pieces[y][x - 1] == -2):
                return False

            # Right
            if x < self.n - 1 and (self.pieces[y][x + 1] == 2 or self.pieces[y][x + 1] == -2):
                return False

            return True
