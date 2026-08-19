"""
Graphe d'adjacence entre provinces -- prérequis pour BYM2 et GNN.

IMPORTANT : on n'a pas de polygones administratifs (shapefile des frontières),
seulement les centroïdes (moyenne des communes). La vraie adjacence BYM2/CAR
utilise normalement "provinces qui partagent une frontière". Ici on approxime
avec une TRIANGULATION DE DELAUNAY sur les centroïdes -- une pratique standard
quand les polygones ne sont pas disponibles, mais qui reste une approximation :
elle peut manquer des voisins réels (provinces allongées) ou en ajouter des
faux (centroïdes proches mais séparés par une autre province entre les deux).
Si un shapefile réel existe dans leishmaniose-mvp, il vaudra mieux le brancher
à la place de ce graphe.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

profile = pd.read_csv("/home/claude/zone_epi_final.csv")
coords = profile[["longitude", "latitude"]].values
provinces = profile["province"].values

tri = Delaunay(coords)

# Extraire les arêtes uniques du maillage triangulaire
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i + 1) % 3]
        edges.add((min(a, b), max(a, b)))

# Filtrer les arêtes trop longues (artefacts de bord de la triangulation, ex: reliant
# le Sud lointain au reste alors qu'il n'y a aucune province entre les deux)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

edge_list = []
for i, j in edges:
    d = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
    edge_list.append((i, j, d))

dists = np.array([e[2] for e in edge_list])
threshold = np.percentile(dists, 95)  # coupe les 5% d'arêtes les plus longues (artefacts)
edge_list_filtered = [(i, j, d) for i, j, d in edge_list if d <= threshold]

print(f"{len(edge_list)} arêtes brutes (Delaunay) -> {len(edge_list_filtered)} après filtrage "
      f"(seuil {threshold:.0f} km)")

# Matrice d'adjacence (pour BYM2) + liste d'arêtes (pour GNN, format edge_index)
n = len(provinces)
W = np.zeros((n, n), dtype=int)
for i, j, d in edge_list_filtered:
    W[i, j] = W[j, i] = 1

n_neighbors = W.sum(axis=1)
print(f"Voisins par province : min={n_neighbors.min()}, max={n_neighbors.max()}, "
      f"moyenne={n_neighbors.mean():.1f}, provinces isolées (0 voisin)={ (n_neighbors==0).sum()}")

np.save("/home/claude/adjacency_matrix.npy", W)
pd.DataFrame(W, index=provinces, columns=provinces).to_csv("/home/claude/adjacency_matrix.csv")

edge_index = np.array([(i, j) for i, j, d in edge_list_filtered] +
                       [(j, i) for i, j, d in edge_list_filtered]).T  # symétrique, format PyG
np.save("/home/claude/edge_index.npy", edge_index)

# ---------------------------------------------------------------------------
# Visualisation du graphe
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
for i, j, d in edge_list_filtered:
    ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], "gray", lw=0.6, alpha=0.6)
sc = ax.scatter(coords[:, 0], coords[:, 1], c=profile["zone_bioclim"], cmap="tab10", s=40, zorder=5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Graphe d'adjacence proxy (Delaunay filtré) — {len(edge_list_filtered)} arêtes, {n} provinces\n"
             "Approximation en l'absence de polygones administratifs")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("/home/claude/graphe_adjacence.png", dpi=130)
print("\nFigure sauvegardée : graphe_adjacence.png")
print("Fichiers sauvegardés : adjacency_matrix.npy/.csv (pour BYM2), edge_index.npy (pour GNN, format PyTorch Geometric)")
