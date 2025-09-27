from player import HumanPlayer, RandomComputerPlayer, MinimaxComputerPlayer


def print_board(board):
    symbols = {"x": "X", "o": "O", "": " "}
    for row in board:
        print(" | ".join(symbols[cell] for cell in row))
        print("-" * 9)
    print()


def get_human_move():
    while True:
        try:
            move = input("Enter your move as 'row,column' (e.g., 1,2): ").strip()
            row, col = map(int, move.split(","))
            if row in range(3) and col in range(3):
                return (row, col)
            else:
                print("Invalid input! Row and column must be between 0 and 2.")
        except ValueError:
            print("Invalid format! Enter your move as 'row,col'.")


def run_simulation(game, player_x, player_o):
    print("Starting Tic Tac Toe simulation!")
    ties = 0
    i = 0
    # while i != 100:
    while True:
        print("\nNew Game!")
        game.play_again()
        print(f"start playing {'x' if game.player_x_turn else 'o'}")
        while game.get_winner() == "":
            print_board(game.board)
            current_player = player_x if game.player_x_turn else player_o
            if isinstance(current_player, HumanPlayer):
                print("Your turn")
                move = get_human_move()
            else:
                move = current_player.get_move(None)
            if game.is_free(move):
                game.move(move)
            else:
                print(f"Invalid move at {move}! Try again.")

        winner = game.get_winner()
        print_board(game.board)

        if winner == "x":
            print("The winner is 'X'!")
            player_x.score += 1
        elif winner == "o":
            print("The winner is 'O'!")
            player_o.score += 1
        else:
            print("It's a tie!")
            ties += 1

        # i += 1
        print(f"Scores: X - {player_x.score}, O - {player_o.score}, Ties - {ties}")

        replay = input("Do you want to play again? (y/n): ").strip().lower()
        if replay != "y":
            print("Thanks for playing!")
            break
