import numpy as np
import matplotlib.pyplot as plt



def sample_reserve_price():
    # 40% chance from [160, 200], 60% from [250, 320]
    if np.random.rand() < 0.4:
        return np.random.uniform(160, 200)
    else:
        return np.random.uniform(250, 320)

# Define multiple opponent strategies
strategy_pool = [
    lambda: np.random.uniform(260, 280),  # conservative
    lambda: np.random.uniform(270, 290),  # neutral
    lambda: np.random.uniform(280, 300),  # aggressive
    lambda: 285.0,                        # fixed
]

def simulate_avg_second_bid(n_opponents: int = 100):
    return np.mean([np.random.choice(strategy_pool)() for _ in range(n_opponents)])

def simulate_one_round(first_bid: float, second_bid: float, avg_second_bid: float, n_turtles: int, player_second_bids: list):
    pnl = 0
    # Sample reserve prices
    reserve_prices = [sample_reserve_price() for _ in range(n_turtles)]

    # Combine bids
    all_first_bids = [first_bid] + [np.random.choice(strategy_pool)() for _ in range(len(player_second_bids))]
    all_second_bids = [second_bid] + player_second_bids

    for reserve in reserve_prices:
        # Determine legal bids
        valid_bids = []
        for i, b in enumerate([first_bid, second_bid] + player_second_bids):
            if b >= reserve and not (200 <= b <= 250):
                valid_bids.append((i, b))
        if not valid_bids:
            continue

        winner = min(valid_bids, key=lambda x: x[1])

        if winner[0] == 0:
            pnl += 320 - winner[1]
        elif winner[0] == 1:
            if second_bid >= avg_second_bid:
                pnl += 320 - second_bid
            else:
                p = ((320 - avg_second_bid) / (320 - second_bid)) ** 3
                pnl += p * (320 - second_bid)
    return pnl

def monte_carlo_simulation(first_bid: float, second_bid: float, n_rounds: int = 10000, n_turtles: int = 100, n_players: int = 100):
    total_pnl = 0
    for _ in range(n_rounds):
        player_second_bids = [np.random.choice(strategy_pool)() for _ in range(n_players)]
        avg_second_bid = np.mean(player_second_bids)
        total_pnl += simulate_one_round(first_bid, second_bid, avg_second_bid, n_turtles, player_second_bids)
    return total_pnl / n_rounds

def plot(second_bids: np.ndarray, results: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.plot(second_bids, results, marker='o')
    plt.title("Expected PNL vs Second Bid (Robust across opponent strategies)")
    plt.xlabel("Second Bid")
    plt.ylabel("Expected PNL")
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    first_bid = 198
    second_bids = np.linspace(260, 300, 41)
    results = []
    best_result = 0
    for sb in second_bids:
        expected_pnl = monte_carlo_simulation(first_bid, sb, n_turtles=100, n_players=100)
        results.append(expected_pnl)
        print(f"Second bid: {sb:.2f}, Expected PNL: {expected_pnl:.2f}")

        if expected_pnl > best_result:
            best_result = expected_pnl
            best_second_bid = sb
            print(f"Best second bid: {best_second_bid:.2f}, Expected PNL: {best_result:.2f}")

    #plot(second_bids, results)
