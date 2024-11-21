def print_board(board):
    symbols = {"x": "X", "o": "O", "": " "}
    for row in board:
        print(" | ".join(symbols[cell] for cell in row))
        print("-" * 9)
    print()


def run_simulation(game, player_x, player_o):
    print("Starting Tic Tac Toe simulation!")
    ties = 0
    while True:
        print("\nNew Game!")
        game.play_again()
        while game.get_winner() == "":
            print_board(game.board)
            current_player = player_x if game.player_x_turn else player_o
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

        print(f"Scores: X - {player_x.score}, O - {player_o.score}, Ties - {ties}")

        replay = input("Do you want to play again? (y/n): ").strip().lower()
        if replay != "y":
            print("Thanks for playing!")
            break
