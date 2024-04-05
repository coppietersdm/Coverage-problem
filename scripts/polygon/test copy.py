import pygame
import sys

pygame.init()

ecran = pygame.display.set_mode((600, 400))

points = [(100, 100), (200, 150), (300, 100), (250, 50)]
couleur_polygone = (0, 255, 0, 128)
couleur_selection = (255, 0, 0)
rayon_selection = 10
point_selectionne = None
polygone_selectionne = False

def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos_souris = pygame.mouse.get_pos()
            polygone_selectionne = True
            for i, point in enumerate(points):
                if distance(pos_souris, point) <= rayon_selection:
                    point_selectionne = i
                    polygone_selectionne = False
                    break

    
        if event.type == pygame.MOUSEBUTTONUP:
            point_selectionne = None
            polygone_selectionne = False

    
        if event.type == pygame.MOUSEMOTION:
            if polygone_selectionne:
            
                for i in range(len(points)):
                    points[i] = (points[i][0] + event.rel[0], points[i][1] + event.rel[1])
            elif point_selectionne is not None:
            
                points[point_selectionne] = pygame.mouse.get_pos()


    ecran.fill((255, 255, 255))

    for x in range(-300, 301, 20):  # Lignes verticales
        pygame.draw.line(ecran, (200, 200, 200), (x + 300, 0), (x + 300, 400))
    for y in range(-200, 201, 20):  # Lignes horizontales
        pygame.draw.line(ecran, (200, 200, 200), (0, y + 200), (600, y + 200))

    # Dessiner les axes x et y
    pygame.draw.line(ecran, (255, 0, 0), (0, 200), (600, 200), 2)  # Axe x (rouge)
    pygame.draw.line(ecran, (0, 0, 255), (300, 0), (300, 400), 2)  # Axe y (bleu)


    pygame.draw.polygon(ecran, couleur_polygone, points, width = 10)


    for i, point in enumerate(points):
        if i == point_selectionne:
            couleur = couleur_selection
        else:
            couleur = (0, 0, 0)
        pygame.draw.circle(ecran, couleur, point, rayon_selection)


    pygame.display.update()
