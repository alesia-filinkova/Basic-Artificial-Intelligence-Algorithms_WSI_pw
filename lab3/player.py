from abc import ABC, abstractmethod
import time
import numpy as np


def build_player(player_config, game):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, player_config)


class Player(ABC):
    def __init__(self, game):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game, config):
        super().__init__(game)
        self.depth = config["depth"]
        print(self.depth)

    def get_move(self, event_position):
        start = time.time()
        self.my_symbol = "x" if self.game.player_x_turn else "o"
        self.opponent_symbol = "x" if self.my_symbol == "o" else "o"

        best_score = float("-inf")
        best_move = None

        for move in self.game.available_moves():
            self.game.move(move)
            score = self.minimax(self.depth, float("-inf"), float("inf"), False)
            self.game.undo_move(move)

            if score > best_score:
                best_score = score
                best_move = move

        end = time.time()
        print(f"Time of one move by minimax: {end - start}")
        return best_move

    def minimax(self, depth, alpha, beta, is_maximizing):
        winner = self.game.get_winner()
        if winner == self.my_symbol:
            return 1
        elif winner == self.opponent_symbol:
            return -1
        elif winner == "t":
            return 0

        if depth == 0:
            return self.evaluate_board()

        if is_maximizing:
            max_eval = float("-inf")
            for move in self.game.available_moves():
                self.game.move(move)
                eval = self.minimax(depth - 1, alpha, beta, False)
                self.game.undo_move(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in self.game.available_moves():
                self.game.move(move)
                eval = self.minimax(depth - 1, alpha, beta, True)
                self.game.undo_move(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break 
            return min_eval

    def evaluate_board(self):
        score = 0
        for row in self.game.board:
            score += self.evaluate_line(row)
        for col in self.game.board.T:
            score += self.evaluate_line(col)
        score += self.evaluate_line(np.diagonal(self.game.board))
        score += self.evaluate_line(np.diagonal(np.fliplr(self.game.board)))
        return score

    def evaluate_line(self, line):
        my_count = np.count_nonzero(line == self.my_symbol)
        opponent_count = np.count_nonzero(line == self.opponent_symbol)

        if my_count > 0 and opponent_count > 0:
            return 0 
        elif my_count == 2:
            return 20**my_count
        elif my_count > 0 or opponent_count == 2:
            return 10**my_count
        elif opponent_count > 0:
            return -(10**opponent_count)
        return 0
