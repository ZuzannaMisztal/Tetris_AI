import copy
import time
from collections import Counter

import pygame
import numpy as np
import random
import more_itertools

pygame.init()

BLOCK_SIZE = 30
BLOCK_FILLING = 28
BLOCK_MARGIN = 1
INFO_WIDTH = 115
HEIGHT = 16
WIDTH = 10
MAP_HEIGHT = HEIGHT * BLOCK_SIZE
MAP_WIDTH = WIDTH * BLOCK_SIZE + INFO_WIDTH
MAP_SIZE = [MAP_WIDTH, MAP_HEIGHT]
CLOCK = pygame.time.Clock()
FPS = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (71, 0, 178)
RED = (255, 0, 0)
GREEN = (0, 172, 23)
ORANGE = (255, 133, 0)
PINK = (255, 133, 155)
AQUA = (0, 255, 255)

COLORS = {
    'white': WHITE,
    'ye': YELLOW,
    'bl': BLUE,
    're': RED,
    'gr': GREEN,
    'or': ORANGE,
    'pi': PINK,
    'aq': AQUA
}


class Tetromino:
    def __init__(self, color, position, squares, max_rotation):
        self.color = color
        self.pos = position
        self.squares = squares
        self.max_rot = max_rotation

    def rotate(self, times=1):
        for _ in range(times):
            for square in self.squares:
                square[0], square[1] = -square[1], square[0]

    def move_down(self, times=1):
        self.pos[1] += times

    def move_right(self, times=1):
        self.pos[0] += times


PIECES = {
    'S': Tetromino('re', [4, 1], [[-1, 0], [0, 0], [0, -1], [1, -1]], 2),
    'Z': Tetromino('gr', [4, 1], [[-1, -1], [0, -1], [0, 0], [1, 0]], 2),
    'J': Tetromino('pi', [4, 1], [[0, -1], [0, 0], [0, 1], [-1, 1]], 4),
    'L': Tetromino('or', [4, 1], [[0, -1], [0, 0], [0, 1], [1, 1]], 4),
    'I': Tetromino('bl', [4, 2], [[0, -2], [0, -1], [0, 0], [0, 1]], 2),
    'O': Tetromino('ye', [4, 0], [[0, 0], [1, 0], [0, 1], [1, 1]], 1),
    'T': Tetromino('aq', [4, 1], [[-1, 0], [0, 0], [1, 0], [0, -1]], 4)
}


DANGER_ZONE = [[3, 2], [4, 2], [5, 2], [4, 3]]


def is_outside_board(x, y):
    return x < 0 or x > WIDTH - 1 or y > HEIGHT - 1


def is_colliding(x, y, block_grid):
    return block_grid[y][x] != 'white'


def blend_tetromino(block_grid, tetromino: Tetromino):
    for square in tetromino.squares:
        x = tetromino.pos[0] + square[0]
        y = tetromino.pos[1] + square[1]
        block_grid[y][x] = tetromino.color
    return block_grid


def clear_lines(block_grid):
    lines_cleared = 0
    for i, row in enumerate(block_grid):
        if np.any([color == 'white' for color in row]):
            continue
        block_grid = np.delete(block_grid, i, axis=0)
        clear_row = np.full((1, WIDTH), 'white')
        block_grid = np.vstack((clear_row, block_grid))
        lines_cleared += 1
    return block_grid, lines_cleared


def settle(block_grid, tetromino: Tetromino, move):
    tetromino.rotate(move[0])
    tetromino.move_right(move[1])
    if not Game.is_valid_position(block_grid, tetromino):
        return None, 0
    while Game.is_valid_position(block_grid, tetromino, adj_y=1):
        tetromino.move_down()
    block_grid = blend_tetromino(block_grid, tetromino)
    block_grid, lines_cleared = clear_lines(block_grid)
    return block_grid, lines_cleared


def find_best_move_greedy(block_grid, tetromino):
    move_list = []
    score_list = []
    for rot in range(tetromino.max_rot):
        for sideways in range(-4, 6):
            move = [rot, sideways]
            result_board, lines_cleared = settle(copy.deepcopy(block_grid), copy.deepcopy(tetromino), move)
            if result_board is not None:
                score = Calculator(result_board, lines_cleared).calculate()
                move_list.append(move)
                score_list.append(score)
    return move_list[score_list.index(max(score_list))]


def find_best_move(block_grid, tetromino, next_tetromino):
    move_list = []
    score_list = []
    for rot in range(tetromino.max_rot):
        for sideways in range(-4, 6):
            move = [rot, sideways]
            result_board, lines_cleared = settle(copy.deepcopy(block_grid), copy.deepcopy(tetromino), move)
            if result_board is not None:
                score_list2 = []
                for rot2 in range(next_tetromino.max_rot):
                    for sideways2 in range(-4, 6):
                        move2 = [rot2, sideways2]
                        result_board2, lines_cleared2 = settle(copy.deepcopy(result_board), copy.deepcopy(next_tetromino), move2)
                        if result_board2 is not None:
                            score = Calculator(result_board2, lines_cleared + lines_cleared2).calculate()
                            score_list2.append(score)
                move_list.append(move)
                score_list.append(max(score_list2))
    return move_list[score_list.index(max(score_list))]


def find_best_move_and_remember(block_grid, tetromino, next_tetromino, remembered, n=4):
    score_list = []
    to_remember = []
    for move in remembered:
        result_board, lines_cleared = settle(copy.deepcopy(block_grid), copy.deepcopy(tetromino), move)
        move_list2 = []
        score_list2 = []
        for rot in range(next_tetromino.max_rot):
            for sideways in range(-4, 6):
                move2 = (rot, sideways)
                result_board2, lines_cleared2 = settle(copy.deepcopy(result_board), copy.deepcopy(next_tetromino), move2)
                if result_board2 is not None:
                    score = Calculator(result_board2, lines_cleared + lines_cleared2).calculate()
                    move_list2.append(move2)
                    score_list2.append(score)
        next_moves_with_scores = Counter(dict((zip(move_list2, score_list2)))).most_common(n)
        unzipped = list(zip(*next_moves_with_scores))
        score_list.append(np.max(unzipped[1]))
        next_moves = list(unzipped[0])
        to_remember.append(next_moves)
    move_index = score_list.index(max(score_list))
    return remembered[move_index], to_remember[move_index]


class Game:
    def __init__(self):
        self.block_grid = np.full((HEIGHT, WIDTH), 'white')
        self.tetromino = self.generate_tetromino()
        self.next_tetromino = self.generate_tetromino()
        self.lines_cleared = 0
        self.game_display = pygame.display.set_mode(MAP_SIZE)
        self.sum_placing_time = 0
        self.blocks_placed = 0
        self.best_next_moves = self.find_initial_moves()

    @property
    def average_placing_time(self):
        if self.blocks_placed > 0:
            return self.sum_placing_time / self.blocks_placed
        return 0

    @staticmethod
    def generate_tetromino():
        return copy.deepcopy(random.choice(list(PIECES.values())))

    @staticmethod
    def is_valid_position(block_grid, tetromino: Tetromino, adj_x=0, adj_y=0, rot=0):
        test_tet = copy.deepcopy(tetromino)
        test_tet.rotate(rot)
        test_tet.move_down(adj_y)
        test_tet.move_right(adj_x)
        for square in test_tet.squares:
            x = test_tet.pos[0] + square[0]
            y = test_tet.pos[1] + square[1]
            if is_outside_board(x, y) or is_colliding(x, y, block_grid):
                return False
        return True

    def run(self):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.lines_cleared, self.average_placing_time
            if not self.is_valid_position(self.block_grid, self.tetromino):
                pygame.quit()
                return self.lines_cleared, self.average_placing_time
            self.draw()
            start = time.time()

            current_move = find_best_move_greedy(self.block_grid, self.tetromino)
            #current_move = find_best_move(self.block_grid, self.tetromino, self.next_tetromino)
            #current_move, self.best_next_moves = find_best_move_and_remember(self.block_grid, self.tetromino, self.next_tetromino, self.best_next_moves, n=4)

            end = time.time()
            self.sum_placing_time += end-start
            self.blocks_placed += 1
            self.block_grid, cleared_lines = settle(self.block_grid, self.tetromino, current_move)
            self.lines_cleared += cleared_lines
            self.tetromino = self.next_tetromino
            self.next_tetromino = self.generate_tetromino()
            CLOCK.tick(FPS)

    def find_initial_moves(self):
        move_list = []
        score_list = []
        for rot in range(self.tetromino.max_rot):
            for sideways in range(-4, 6):
                move = (rot, sideways)
                result_board, lines_cleared = settle(copy.deepcopy(self.block_grid), copy.deepcopy(self.tetromino), move)
                if result_board is not None:
                    score = Calculator(result_board, lines_cleared).calculate()
                    move_list.append(move)
                    score_list.append(score)
        move_to_score = Counter(dict(zip(move_list, score_list)))
        best_moves = [move for move, score in move_to_score.most_common(4)]
        return best_moves

    def draw(self):
        self.game_display.fill(BLACK)
        self.draw_board()
        self.draw_tetromino()
        self.print_score()
        pygame.display.update()

    def draw_board(self):
        pygame.draw.rect(self.game_display, WHITE, [0, 0, WIDTH * BLOCK_SIZE, HEIGHT * BLOCK_SIZE])
        for y, row in enumerate(self.block_grid):
            for x, color in enumerate(row):
                if color != 'white':
                    self.draw_rect(x, y, color)

    def draw_rect(self, x, y, color):
        pygame.draw.rect(self.game_display, BLACK,
                         [x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
        pygame.draw.rect(self.game_display, COLORS[color],
                         [x * BLOCK_SIZE + BLOCK_MARGIN,
                          y * BLOCK_SIZE + BLOCK_MARGIN,
                          BLOCK_FILLING, BLOCK_FILLING])

    def draw_tetromino(self):
        tet = self.tetromino
        for square in tet.squares:
            self.draw_rect(tet.pos[0] + square[0], tet.pos[1] + square[1], tet.color)

    def print_score(self):
        font = pygame.font.SysFont('monospace', 40)
        text = font.render(str(self.lines_cleared), True, WHITE)
        self.game_display.blit(text, [310, 20])


class Calculator:
    def __init__(self, block_grid, lines_cleared):
        self.block_grid = block_grid
        self.lines_cleared = lines_cleared

    def calculate(self):
        score = 50 * self.lines_cleared
        score += -2 * self.bumpiness()
        score += -30 * self.holes_simple()
        if self.is_it_game_over():
            score += -2000
        return score

    def bumpiness(self):
        column_heights = [self.column_height(column) for column in self.block_grid.T]
        result = sum(abs(pair[0] - pair[1]) for pair in more_itertools.pairwise(column_heights))
        if column_heights[1] > column_heights[0]:
            result += column_heights[1] - column_heights[0]
        if column_heights[-2] > column_heights[-1]:
            result += column_heights[-2] - column_heights[-1]
        return result

    def holes_simple(self):
        return sum(self.column_holes(column) for column in self.block_grid.T)

    def is_it_game_over(self):
        return np.any([self.block_grid[y][x] != 'white' for x, y in DANGER_ZONE])

    def holes_by_height(self):
        return sum(self.column_holes_by_height(column) for column in self.block_grid.T)

    def aggregate_height(self):
        return sum(self.column_height(column) for column in self.block_grid.T)

    @staticmethod
    def column_height(column):
        height = HEIGHT
        for color in column:
            if color != 'white':
                break
            height -= 1
        return height

    def column_holes(self, column):
        highest = self.column_height(column)
        return sum(1 for i in range(HEIGHT - highest, HEIGHT) if (column[i] == 'white'))

    def column_holes_by_height(self, column):
        highest = self.column_height(column)
        result = 0
        for i in range(highest):
            if column[i] != 'white':
                result += (i + 1)**3
        return result


if __name__ == '__main__':
    game = Game()
    score_total, avg_time = game.run()
    print(f"Deleted lines = {score_total}")
    print(f"Average time taken to place tetromino in seconds = {avg_time}")
