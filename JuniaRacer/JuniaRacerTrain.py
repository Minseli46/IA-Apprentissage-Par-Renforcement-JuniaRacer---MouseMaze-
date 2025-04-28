import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import random
import math
import numpy as np
import time
from drivers.c3po import *

pygame.init()

# Mode d'entraînement sans affichage graphique
TRAINING_MODE = True

# Variables initiales
size = width, height = 1600, 900
white = (255, 255, 255)
Color_line = (255, 0, 0)
FPS = 120
maxspeed = 10

# Chargement des images
white_small_car = pygame.image.load('Images/Sprites/white_small.png')
bg = pygame.image.load('bg73.png')
bg4 = pygame.image.load('bg43.png')

# Fonctions utilitaires
def calculateDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def rotation(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def move(point, angle, unit):
    x, y = point
    rad = math.radians(-angle % 360)
    x += unit * math.sin(rad)
    y += unit * math.cos(rad)
    return x, y

class Car:
    def __init__(self):
        self.x = 120
        self.y = 480
        self.center = (self.x, self.y)
        self.height = 35
        self.width = 17
        self.d = (self.x - (self.width / 2), self.y - (self.height / 2))
        self.c = (self.x + self.width - (self.width / 2), self.y - (self.height / 2))
        self.b = (self.x + self.width - (self.width / 2), self.y + (self.height - self.height / 2))
        self.a = (self.x - (self.width / 2), self.y + (self.height - self.height / 2))
        self.velocity = 0
        self.acceleration = 0
        self.angle = 180
        self.collided = False
        self.car_image = white_small_car
        self.last_x = self.x
        self.c1 = (0, 0)
        self.c2 = (0, 0)
        self.c3 = (0, 0)
        self.c4 = (0, 0)
        self.c5 = (0, 0)
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0
        self.d5 = 0
        self.showlines = True
        self.distance_traveled = 0

    def set_accel(self, accel):
        self.acceleration = accel

    def rotate(self, rot):
        self.angle = (self.angle + rot) % 360

    def update(self):
        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > maxspeed:
                self.velocity = maxspeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92

        old_x, old_y = self.x, self.y
        self.last_x = self.x
        self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
        self.center = (self.x, self.y)
        self.distance_traveled += calculateDistance(old_x, old_y, self.x, self.y)

        self.d = (self.x - (self.width / 2), self.y - (self.height / 2))
        self.c = (self.x + self.width - (self.width / 2), self.y - (self.height / 2))
        self.b = (self.x + self.width - (self.width / 2), self.y + (self.height - self.height / 2))
        self.a = (self.x - (self.width / 2), self.y + (self.height - self.height / 2))

        self.a = rotation((self.x, self.y), self.a, math.radians(self.angle))
        self.b = rotation((self.x, self.y), self.b, math.radians(self.angle))
        self.c = rotation((self.x, self.y), self.c, math.radians(self.angle))
        self.d = rotation((self.x, self.y), self.d, math.radians(self.angle))

        self.c1 = move((self.x, self.y), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a != 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a == 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, -1)

        self.c2 = move((self.x, self.y), self.angle + 45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a != 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle + 45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a == 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle + 45, -1)

        self.c3 = move((self.x, self.y), self.angle - 45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a != 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle - 45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a == 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle - 45, -1)

        self.c4 = move((self.x, self.y), self.angle + 90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a != 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle + 90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a == 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle + 90, -1)

        self.c5 = move((self.x, self.y), self.angle - 90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a != 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle - 90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a == 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle - 90, -1)

        self.d1 = int(calculateDistance(self.center[0], self.center[1], self.c1[0], self.c1[1]))
        self.d2 = int(calculateDistance(self.center[0], self.center[1], self.c2[0], self.c2[1]))
        self.d3 = int(calculateDistance(self.center[0], self.center[1], self.c3[0], self.c3[1]))
        self.d4 = int(calculateDistance(self.center[0], self.center[1], self.c4[0], self.c4[1]))
        self.d5 = int(calculateDistance(self.center[0], self.center[1], self.c5[0], self.c5[1]))

    def draw(self, display):
        # La fonction draw n'est plus utilisée en mode entraînement headless
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle - 180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = (self.x, self.y)
        display.blit(rotated_image, rect_rotated_image)
        if self.showlines:
            pygame.draw.line(display, Color_line, (self.x, self.y), self.c1, 2)
            pygame.draw.line(display, Color_line, (self.x, self.y), self.c2, 2)
            pygame.draw.line(display, Color_line, (self.x, self.y), self.c3, 2)
            pygame.draw.line(display, Color_line, (self.x, self.y), self.c4, 2)
            pygame.draw.line(display, Color_line, (self.x, self.y), self.c5, 2)

    def collision(self):
        return (bg4.get_at((int(self.a[0]), int(self.a[1]))).a == 0 or
                bg4.get_at((int(self.b[0]), int(self.b[1]))).a == 0 or
                bg4.get_at((int(self.c[0]), int(self.c[1]))).a == 0 or
                bg4.get_at((int(self.d[0]), int(self.d[1]))).a == 0)

    def resetPosition(self):
        self.x = 120
        self.y = 480
        self.angle = 180
        self.velocity = 0
        self.acceleration = 0
        self.last_x = self.x
        self.distance_traveled = 0

# Création de la fenêtre en mode caché si TRAINING_MODE est activé
if TRAINING_MODE:
    gameDisplay = pygame.display.set_mode(size, flags=pygame.HIDDEN)
else:
    gameDisplay = pygame.display.set_mode(size)
clock = pygame.time.Clock()
car = Car()

def train_car(episodes=1000, max_steps=10000):
    global EPSILON
    EPSILON_DECAY = (EPSILON_START - EPSILON_END) / episodes  # Décroissance sur les épisodes
    print("Début de l'entraînement...")
    for episode in range(episodes):
        car.resetPosition()
        episode_reward = 0
        tour_count = 0
        EPSILON = max(EPSILON_END, EPSILON_START - episode * EPSILON_DECAY)
        
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_q_table()
                    pygame.quit()
                    sys.exit()

            # Récupération de l'action depuis le driver
            action = drive(car.d1, car.d2, car.d3, car.d4, car.d5, car.velocity, car.acceleration)
            if action == LEFT5:
                car.rotate(-5)
            elif action == RIGHT5:
                car.rotate(5)
            elif action == ACCELERATE:
                car.set_accel(0.2)
            elif action == BRAKE:
                car.set_accel(-0.2)
            else:
                car.set_accel(0)

            car.update()

            # Calcul de la récompense
            centering_bonus = -abs(car.d4 - car.d5) * 0.5
            front_safety = car.d1 * 0.3 if car.d1 > 15 else -50
            diagonal_safety = (min(car.d2, car.d3) * 0.2 if min(car.d2, car.d3) > 10 else -30)
            reward = (car.velocity * 2 + car.distance_traveled * 0.01 +
                      centering_bonus + front_safety + diagonal_safety)

            if car.collision():
                reward = -1000
                car.resetPosition()
                print(f"Collision à l'étape {step}, épisode {episode + 1}")

            # Vérifier la complétion d'un tour (exemple d'une condition simplifiée)
            if car.last_x < 1400 and car.x >= 1400 and 400 < car.y < 500:
                tour_count += 1
                reward += 200
                print(f"Tour {tour_count} complété dans l'épisode {episode + 1}")
                if tour_count == 3:
                    print(f"Épisode {episode + 1}: 3 tours complétés")
                    break

            update_reward(reward)
            episode_reward += reward

            # En mode entraînement headless, on n'affiche pas le rendu
            if not TRAINING_MODE:
                gameDisplay.blit(bg, (0, 0))
                car.draw(gameDisplay)
                font = pygame.font.Font(None, 36)
                text = font.render(f"Ép: {episode + 1}/{episodes}, R: {episode_reward:.1f}, T: {tour_count}, "
                                  f"Epsilon: {EPSILON:.3f}", True, (255, 255, 255))
                gameDisplay.blit(text, (10, 10))
                pygame.display.update()
                clock.tick(60)
        
        print(f"Épisode {episode + 1}/{episodes}, Récompense: {episode_reward:.1f}, Tours: {tour_count}, Epsilon: {EPSILON:.3f}")
    save_q_table()

if __name__ == "__main__":
    setup()
    train_car(episodes=3000, max_steps=10000)
