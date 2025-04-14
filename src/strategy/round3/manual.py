import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

# Define multiple opponent strategies with long-tail behavior
def generate_long_tail_bid(low, high, alpha=2.5):
    """ Generate a long-tail distribution bid using the pareto distribution and truncate it within [low, high] """
    bid = low + (high - low) * (np.random.pareto(alpha) / (np.random.pareto(alpha) + 1))
    return np.round(bid)  # Round to nearest integer

# Define multiple opponent strategies
strategy_pool = [
    lambda: np.random.uniform(260, 280),  # conservative
    lambda: np.random.uniform(270, 290),  # neutral
    lambda: np.random.uniform(280, 300),  # aggressive
    lambda: 285.0,                        # fixed
]

def sample_reserve_price():
    """ Sample a reserve price for the turtles """
    if np.random.rand() < 0.4:
        return generate_long_tail_bid(160, 200)  # 40% chance for the [160, 200] range with long-tail distribution
    else:
        return generate_long_tail_bid(250, 320)  # 60% chance for the [250, 320] range with long-tail distribution

def simulate_player_bids(n_players: int, first_bid_options: list, second_bid_options: list):
    """ Simulate first and second bids from multiple players based on different strategies """
    first_bids = np.random.choice(first_bid_options, n_players)
    second_bids = np.random.choice(second_bid_options, n_players)
    return first_bids, second_bids

def simulate_one_round(first_bids, second_bids, n_turtles: int, avg_second_bid: float):
    pnl = 0
    # Sample reserve prices for turtles
    reserve_prices = [sample_reserve_price() for _ in range(n_turtles)]

    for reserve in reserve_prices:
        # Determine legal bids for this turtle (first and second bids of players)
        valid_bids = []
        for i, (f_bid, s_bid) in enumerate(zip(first_bids, second_bids)):
            if f_bid >= reserve and not (200 <= f_bid <= 250):  # Valid first bid
                valid_bids.append((i, f_bid, 'first'))  # Store player index and bid type
            if s_bid >= reserve and not (200 <= s_bid <= 250):  # Valid second bid
                valid_bids.append((i, s_bid, 'second'))  # Store player index and bid type

        if not valid_bids:
            continue

        # Select the lowest valid bid
        winner = min(valid_bids, key=lambda x: x[1])

        if winner[2] == 'first':
            pnl += 320 - winner[1]  # Player wins with first bid
        elif winner[2] == 'second':
            if winner[1] >= avg_second_bid:
                pnl += 320 - winner[1]
            else:
                p = ((320 - avg_second_bid) / (320 - winner[1])) ** 3
                pnl += p * (320 - winner[1])
    return pnl

def monte_carlo_simulation(n_players: int, n_turtles: int, first_bid_options: list, second_bid_options: list, n_rounds: int = 10000):
    total_pnl = 0
    for _ in range(n_rounds):
        # Simulate player bids (first and second)
        first_bids, second_bids = simulate_player_bids(n_players, first_bid_options, second_bid_options)

        # Calculate the average second bid
        avg_second_bid = np.mean(second_bids)

        # Calculate PNL for this round
        total_pnl += simulate_one_round(first_bids, second_bids, n_turtles, avg_second_bid)
    
    return total_pnl / n_rounds

def parallel_monte_carlo_simulation(combination):
    first_bid, second_bid, n_players, n_turtles, n_rounds = combination
    return monte_carlo_simulation(n_players, n_turtles, [first_bid], [second_bid], n_rounds)

def find_best_bid_combination(n_players: int, n_turtles: int, first_bid_options: list, second_bid_options: list, n_rounds: int = 1000):
    best_pnl = -np.inf
    best_combination = (None, None)
    
    # Create all combinations of first_bid and second_bid
    combinations = [(first_bid, second_bid, n_players, n_turtles, n_rounds) for first_bid in first_bid_options for second_bid in second_bid_options]

    # Parallelize the Monte Carlo simulations
    with Pool() as pool:
        results = list(tqdm(pool.imap(parallel_monte_carlo_simulation, combinations), total=len(combinations)))

    # Find the best combination
    for idx, pnl in enumerate(results):
        first_bid, second_bid, _, _, _ = combinations[idx]
        print(f"First bid: {first_bid}, Second bid: {second_bid}, Average PNL: {pnl}")
        if pnl > best_pnl:
            best_pnl = pnl
            best_combination = (first_bid, second_bid)
    
    print(f"Best combination: First bid = {best_combination[0]}, Second bid = {best_combination[1]}, PNL = {best_pnl}")

if __name__ == "__main__":
    # Define the range of possible first and second bids
    first_bid_options = np.arange(160, 201, 1).tolist() + np.arange(250, 321, 1).tolist()  # Integer bids between 160-200 and 250-320
    second_bid_options = np.arange(260, 301, 1).tolist()  # Integer bids between 260-300
    
    # Simulate and find the best bid combination
    n_players = 1000  # Number of players
    n_turtles = 1000  # Number of turtles
    find_best_bid_combination(n_players, n_turtles, first_bid_options, second_bid_options, n_rounds=1000)