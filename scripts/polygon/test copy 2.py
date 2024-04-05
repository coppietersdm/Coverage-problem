import pygame
import sys



pygame.init()

M,N = 1000,1000

ecran = pygame.display.set_mode((M, N))

font = pygame.font.SysFont(None, 20)  # Create a font object for displaying text

points = [(100, 100), (100, -100), (-100, -100), (-100, 100)]

for i in range(len(points)):
    points[i] = (points[i][0] + M//2, points[i][1] + N//2)
    
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

    for x in range(0, M, 100):  # Lignes verticales
        pygame.draw.line(ecran, (200, 200, 200), (x, 0), (x, N))
    for y in range(0, N, 100):  # Lignes horizontales
        pygame.draw.line(ecran, (200, 200, 200), (0, y), (M, y))

    # Dessiner les axes x et y
    pygame.draw.line(ecran, (255, 0, 0), (0, N//2), (M, N//2), 2)  # Axe x (rouge)
    pygame.draw.line(ecran, (0, 0, 255), (M/2, 0), (M//2, N), 2)  # Axe y (bleu)


    pygame.draw.polygon(ecran, couleur_polygone, points, width = 10)


    for i, point in enumerate(points):
        if i == point_selectionne:
            couleur = couleur_selection
        else:
            couleur = (0, 0, 0)
        pygame.draw.circle(ecran, couleur, point, rayon_selection)

        # Create text surface with coordinates and render it next to the point
        coord_text = font.render(f"({(point[0]-M//2)/100}, {(point[1]-N//2)/100})", True, (0, 0, 0))  # Black text
        text_rect = coord_text.get_rect(center=point)  # Center the text next to the point
        ecran.blit(coord_text, text_rect)


    pygame.display.update()
