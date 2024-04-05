import pygame

# Initialiser Pygame
pygame.init()

# Définir la taille de l'écran
ecran = pygame.display.set_mode((600, 400))

# Créer une liste de points pour le polygone
points = [(100, 100), (200, 150), (300, 100), (250, 50)]

# Définir la couleur du polygone
couleur_polygone = (0, 255, 0)

# Définir la couleur du point sélectionné
couleur_selection = (255, 0, 0)

# Définir le rayon du point sélectionné
rayon_selection = 10

# Variable pour stocker le point sélectionné (initialisé à None)
point_selectionne = None

# Fonction pour calculer la distance entre deux points
def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

# Boucle principale
while True:

    # Gérer les événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Détecter la sélection d'un point
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos_souris = pygame.mouse.get_pos()
            for i, point in enumerate(points):
                if distance(pos_souris, point) <= rayon_selection:
                    point_selectionne = i
                    break
        
        # Détecter le relâchement du bouton de la souris
        if event.type == pygame.MOUSEBUTTONUP:
            point_selectionne = None  # Relâchement du bouton : aucun point sélectionné

        # Déplacer le point sélectionné
        if event.type == pygame.MOUSEMOTION:
            if point_selectionne is not None:
                points[point_selectionne] = pygame.mouse.get_pos()

        # Déplacer le point sélectionné
        if event.type == pygame.MOUSEMOTION:
            if point_selectionne is not None:  # Vérifier si un point est sélectionné avant de le déplacer
                points[point_selectionne] = pygame.mouse.get_pos()


    # Remplir l'écran de blanc
    ecran.fill((255, 255, 255))

    # Dessiner le polygone
    pygame.draw.polygon(ecran, couleur_polygone, points)

    # Dessiner les points
    for i, point in enumerate(points):
        if i == point_selectionne:
            couleur = couleur_selection
        else:
            couleur = (0, 0, 0)
        pygame.draw.circle(ecran, couleur, point, rayon_selection)

    # Mettre à jour l'écran
    pygame.display.update()
