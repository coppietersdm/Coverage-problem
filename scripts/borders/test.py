import cv2

# Ouvrir l'image
image = cv2.imread("borders/test3.png")

# Convertir l'image en niveaux de gris
image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre de seuil pour binariser l'image
image_binaire = cv2.threshold(image_gris, 200, 255, cv2.THRESH_BINARY)[1]

# Détecter les contours
contours, _ = cv2.findContours(image_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

import numpy as np
import matplotlib.pyplot as plt
data = (np.reshape(contours[0],(-1,2)).T)
plt.plot(data[0], data[1], 'r')

data = (np.reshape(contours[1],(-1,2)).T)
plt.plot(data[0], data[1], 'b')

plt.show()

# Afficher les contours
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Afficher l'image
cv2.imshow("Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()