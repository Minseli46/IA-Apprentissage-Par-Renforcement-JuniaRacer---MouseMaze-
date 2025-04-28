import numpy as np
from typing import List, Optional
from gymnasium.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

def is_valid(board: List[List[str]], max_size: int) -> bool:
    """Vérifie si un chemin valide existe du départ à l'arrivée."""
    frontier, discovered = [(0, 0)], set()
    while frontier:
        r, c = frontier.pop()
        if (r, c) in discovered:
            continue
        discovered.add((r, c))
        for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            r_new, c_new = r + x, c + y
            if 0 <= r_new < max_size and 0 <= c_new < max_size:
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """Génère une carte aléatoire avec une garantie de chemin viable."""
    np_random, _ = seeding.np_random(seed)
    while True:
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0], board[-1][-1] = "S", "G"
        if is_valid(board, size):
            return ["".join(row) for row in board]
