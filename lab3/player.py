from abc import ABC, abstractmethod

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
        # todo: lab3 - load pruning depth from config
        self.depth = config.get("depth", 3)  # Domyślna głębokość to 3

    def get_move(self, event_position):
        # todo: lab3 - implement algorithm
        # raise NotImplementedError
        best_move = self.minimax(self.game, self.depth, float('-inf'), float('inf'), True)
        print(self.game.available_moves()[best_move])
        return self.game.available_moves()[best_move]

    def minimax(self, game, depth, alpha, beta, is_maximizing):
        # Sprawdzenie, czy gra została zakończona
        possible_moves = self.game.available_moves()
        winner = self.game.get_winner()
        if winner == "x":  # Gracz maksymalizujący (minimax) wygrał
            return 1  # Wartość dla maksymalizującego
        elif winner == "o":  # Gracz minimalizujący wygrał
            return -1  # Wartość dla minimalizującego
        elif winner == "t":  # Remis
            return 0  # Wartość remisu

        if depth == 0:  # Jeśli osiągnięto maksymalną głębokość
            return 0  # Wartość neutralna (można również dodać heurystykę)

        if is_maximizing:
            max_eval = float('-inf')
            for move in possible_moves:
                self.game.move(move)  # Wykonaj ruch
                eval = self.minimax(self.game.board, depth - 1, alpha, beta, False)
                self.game.undo_move(move)  # Cofnij ruch
                max_eval = max(max_eval, eval)
                # alpha = max(alpha, eval)
                # if beta <= alpha:  # Odcinanie β
                #     break
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                self.game.move(move)  # Wykonaj ruch
                eval = self.minimax(self.game.board, depth - 1, alpha, beta, True)
                self.game.undo_move(move)  # Cofnij ruch
                min_eval = min(min_eval, eval)
                # beta = min(beta, eval)
                # if beta <= alpha:  # Odcinanie α
                #     break
            return min_eval
