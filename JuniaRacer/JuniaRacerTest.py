import sys
import os
import math
import pygame
import numpy as np
from drivers.c3po import setup, drive, LEFT5, RIGHT5, ACCELERATE, BRAKE, update_reward,EPSILON,ALPHA

# Initialisation de Pygame et configuration de l'affichage
pygame.init()
size = width, height = 1600, 900
gameDisplay = pygame.display.set_mode(size)
pygame.display.set_caption("Test Agent - Q-Learning")
clock = pygame.time.Clock()
FPS = 60

# Chargement des images nécessaires
white_small_car = pygame.image.load('Images/Sprites/white_small.png')
bg = pygame.image.load('bg73.png')
bg4 = pygame.image.load('bg43.png')

# Fonctions utilitaires (mêmes que dans train_qlearning.py)
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

# Définition de la classe Car identique à celle utilisée en entraînement
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
        maxspeed = 10
        # Mise à jour de la vitesse
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

        # Rotation des coins
        self.a = rotation((self.x, self.y), self.a, math.radians(self.angle))
        self.b = rotation((self.x, self.y), self.b, math.radians(self.angle))
        self.c = rotation((self.x, self.y), self.c, math.radians(self.angle))
        self.d = rotation((self.x, self.y), self.d, math.radians(self.angle))

        # Calcul des points de détection (les "capteurs")
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
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle - 180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = (self.x, self.y)
        display.blit(rotated_image, rect_rotated_image)
        if self.showlines:
            pygame.draw.line(display, (255, 0, 0), (self.x, self.y), self.c1, 2)
            pygame.draw.line(display, (255, 0, 0), (self.x, self.y), self.c2, 2)
            pygame.draw.line(display, (255, 0, 0), (self.x, self.y), self.c3, 2)
            pygame.draw.line(display, (255, 0, 0), (self.x, self.y), self.c4, 2)
            pygame.draw.line(display, (255, 0, 0), (self.x, self.y), self.c5, 2)

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

# Chargement de la Q-table et initialisation du driver
setup()  # Charge la Q-table depuis le fichier sauvegardé

EPSILON = 0  # Désactivation de l'exploration
ALPHA = 0  # Désactivation de l'apprentissage

# Création de l'agent (voiture)
car = Car()

print("Démarrage du test de l'agent. Fermez la fenêtre pour arrêter.")

# Boucle principale de test
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Récupération de l'action de l'agent à partir des capteurs et de l'état actuel
    action = drive(car.d1, car.d2, car.d3, car.d4, car.d5, car.velocity, car.acceleration)

    # Application de l'action retournée par le driver
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

    # En cas de collision, on affiche un message et on réinitialise la position
    if car.collision():
        print("Collision détectée ! Réinitialisation de la position.")
        car.resetPosition()

    # Affichage
    gameDisplay.blit(bg, (0, 0))
    car.draw(gameDisplay)
    pygame.display.update()

    clock.tick(FPS)

pygame.quit()
sys.exit()
