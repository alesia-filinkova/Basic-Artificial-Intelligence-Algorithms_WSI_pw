import argparse
import pathlib

import numpy as np
import pandas as pd
from solution_utils import generate_solution, decode_solution, evaluate_solution
from genetic_algorythm import EvolutionAlgorithm
from graphic import graphic

MINI_CITIES_NUM = 25


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cities-path",
        type=pathlib.Path,
        required=True,
        help="Path to cities csv file",
    )
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="mini",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Łomża")
    parser.add_argument("--finish", type=str, default="Częstochowa")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[
        ~((data.index == args.finish) | (data.index == args.start))
    ]
    if args.problem_size == "mini":
        city_names = (
            [args.start]
            + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist()
            + [args.finish]
        )
    else:
        city_names = (
            [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]
        )

    return data[city_names].loc[city_names]


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)
    print(generate_solution(data))
    print(decode_solution(data, generate_solution(data)))
    distances_r = evaluate_solution(data, generate_solution(data))
    print(distances_r)
    ea = EvolutionAlgorithm(data)
    best_route = ea.run()
    print(best_route)
    best_route_cities = decode_solution(data, best_route)
    total_distance = evaluate_solution(data, best_route)
    print("Best route:", best_route_cities)
    print("Best route length:", total_distance)
    # graphic(best_route_cities)


if __name__ == "__main__":
    main()
