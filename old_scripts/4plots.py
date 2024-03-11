import matplotlib.pyplot as plt

# Créer une figure et une grille de subplots
fig, axes = plt.subplots(2, 2)

# Tracer des courbes dans chaque subplot
axes[0, 0].plot([1, 2, 3, 4], [5, 6, 7, 8], label="Courbe 1")
axes[0, 1].plot([1, 2, 3, 4], [9, 10, 11, 12], label="Courbe 2")
axes[1, 0].plot([1, 2, 3, 4], [13, 14, 15, 16], label="Courbe 3")
axes[1, 1].plot([1, 2, 3, 4], [17, 18, 19, 20], label="Courbe 4")

# Ajouter des titres et des étiquettes d'axe
axes[0, 0].set_title("Figure 1")
axes[0, 1].set_title("Figure 2")
axes[1, 0].set_title("Figure 3")
axes[1, 1].set_title("Figure 4")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")

# Ajuster la disposition des subplots
plt.tight_layout()

# Afficher la figure
plt.show()
