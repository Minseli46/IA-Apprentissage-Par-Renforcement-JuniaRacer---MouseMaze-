import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.utils import seeding
from os import path

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

class MouseMazeEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}
    
    def __init__(self, render_mode=None, size=8, is_slippery=True):
        self.size = size
        self.nrow, self.ncol = size, size
        self.is_slippery = is_slippery
        self.reward_range = (-100, 500)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "position": spaces.Discrete(size * size),
            "map": spaces.Box(low=0, high=3, shape=(size * size,), dtype=np.uint8)
        })
        
        self.render_mode = render_mode
        self.np_random = None
        
        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (self.window_size[0] // self.ncol, self.window_size[1] // self.nrow)
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.reward_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def _generate_fixed_map(self):
        """
        Crée une carte fixe.
        Les valeurs sont codées ainsi :
          0 : Case vide (Frozen)
          1 : Trou (Hole)
          2 : Bonbon (Candy)
          3 : Arrivée/Goal
        La carte fixe est définie manuellement.
        """
        board = np.zeros((self.size, self.size), dtype=np.uint8)
        # Définir la case goal en bas à droite
        board[self.size - 1, self.ncol - 1] = 3
        
        # Positionner les bonbons (candies) à des emplacements fixes
        candy_positions = [(0, 3), (2, 2), (4, 5), (5, 1)]
        for pos in candy_positions:
            board[pos] = 2
        
        # Positionner les trous à des emplacements fixes
        hole_positions = [(1, 5), (3, 3), (6, 2)]
        for pos in hole_positions:
            board[pos] = 1
        
        self.total_candies = len(candy_positions)
        return board

    def reset(self, seed=None, options=None):
        # Utilise une carte fixe au lieu d'une carte aléatoire.
        self.np_random, _ = seeding.np_random(seed)
        self.desc = self._generate_fixed_map()
        # Initialiser le suivi des cases visitées
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        # Réinitialiser le compteur de bonbons collectés
        self.candies_collected = 0
        # Position de départ fixée à (0,0)
        self.s = 0  
        self.visited[0, 0] = True
        self.lastaction = None
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"prob": 1}

    def _get_obs(self):
        return {"position": self.s, "map": self.desc.flatten()}

    def _move(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    def step(self, a):
        # Position actuelle
        row, col = self.s // self.ncol, self.s % self.ncol
        
        # Effet glissant si applicable
        if self.is_slippery:
            probs = [1/3] * 3
            actions = [(a - 1) % 4, a, (a + 1) % 4]
            a = self.np_random.choice(actions, p=probs)
        
        # Calcul de la nouvelle position
        new_row, new_col = self._move(row, col, a)
        new_state = new_row * self.ncol + new_col
        
        # Pénalité pour revisiter une case déjà visitée
        extra_penalty = 0
        if self.visited[new_row, new_col]:
            extra_penalty = -0.1
        else:
            self.visited[new_row, new_col] = True
        
        # Coût fixe par pas (pour encourager le chemin le plus court)
        step_penalty = -0.005

        # Détermine le type de case sur la carte
        cell = self.desc[new_row, new_col]
        
        if cell == 1:  # Trou (Hole)
            base_reward, terminated = -100, True
        elif cell == 2:  # Bonbon (Candy)
            # Calcul du reward progressif
            # Si c'est le n-ième bonbon collecté, reward = 20 + 5*(n-1)
            self.candies_collected += 1
            base_reward = 20 + 5 * (self.candies_collected - 1)
            # Si tous les bonbons sont collectés, ajouter un bonus de +100
            if self.candies_collected == self.total_candies:
                base_reward += 100
            terminated = False
            # Marquer le bonbon comme consommé
            self.desc[new_row, new_col] = 0
        elif cell == 3:  # Arrivée / Goal
            # Si l'agent n'a pas collecté tous les bonbons, il est pénalisé
            if self.candies_collected < self.total_candies:
                base_reward, terminated = -100, True
            else:
                base_reward, terminated = 500, True
        else:  # Case vide (Frozen)
            base_reward, terminated = 0, False
        
        # Calcul de la récompense finale
        reward = base_reward + step_penalty + extra_penalty
        
        self.s = new_state
        self.lastaction = int(a)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, {"prob": 1/3 if self.is_slippery else 1}

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise ImportError('pygame is not installed, run `pip install "gymnasium[toy-text]"`')
        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Mouse Maze")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.reward_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.reward_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [pygame.transform.scale(pygame.image.load(f), self.cell_size) for f in elfs]
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                self.window_surface.blit(self.ice_img, pos)
                tile = self.desc[y, x]
                if tile == 1:
                    self.window_surface.blit(self.hole_img, pos)
                elif tile == 2:
                    self.window_surface.blit(self.reward_img, pos)
                elif tile == 3:
                    self.window_surface.blit(self.goal_img, pos)
                elif (y, x) == (0, 0) and tile == 0:
                    self.window_surface.blit(self.start_img, pos)
                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]
        self.window_surface.blit(elf_img, cell_rect)
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))

    def close(self):
        if self.window_surface is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
