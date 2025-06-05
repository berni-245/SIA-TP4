import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.Kohonen import KohonenNet

def plot_activation_map(activations: np.ndarray):
    k = activations.shape[0]
    counts = np.zeros((k, k), dtype=int)
    label_dict = [[{} for _ in range(k)] for _ in range(k)]

    for i in range(k):
        for j in range(k):
            act = activations[i][j]
            counts[i, j] = act["total_activations"]
            for key, value in act.items():
                if key != "total_activations" and value > 0:
                    label_dict[i][j][key] = value

    # Crear colormap y normalización para luminancia
    cmap = plt.get_cmap('magma')
    norm = plt.Normalize(vmin=counts.min(), vmax=counts.max()) # pyright: ignore[reportPrivateImportUsage]

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(counts, annot=False, cmap=cmap, cbar_kws={"label": "Entries amount"})

    # Agregar anotaciones con contraste ajustado
    for i in range(k):
        for j in range(k):
            labels = label_dict[i][j]
            if labels:
                sorted_labels = sorted(labels.items(), key=lambda x: -x[1])
                text = "\n".join(f"{label}: {cnt}" for label, cnt in sorted_labels)

                # Obtener color de fondo y calcular luminancia
                rgba = cmap(norm(counts[i, j]))
                r, g, b, _ = rgba
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'black' if luminance > 0.5 else 'white'

                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=text_color, fontsize=8)

    plt.title("Final entries per neuron")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.tight_layout()
    plt.show()

def plot_average_euclidean_distance_of_weights(kn: KohonenNet):
    k = kn.k
    avg_dist_matrix = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            neighbors = kn._get_neighbors(i, j)
            neighbors.remove((i, j))
            if not neighbors:
                avg_dist_matrix[i, j] = 0
                continue
            center_weight = kn.weights[i, j]
            dists = [np.linalg.norm(center_weight - kn.weights[x, y]) for x, y in neighbors]
            avg_dist_matrix[i, j] = np.mean(dists)

    plt.figure(figsize=(8, 6))
    plt.title("Average Euclidean Distance to Neighbors")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    c = plt.imshow(avg_dist_matrix, cmap="gray", origin="upper")

    # Añadir índices correctos
    plt.xticks(ticks=np.arange(k), labels=np.arange(k)) # pyright: ignore[reportArgumentType]
    plt.yticks(ticks=np.arange(k), labels=np.arange(k)) # pyright: ignore[reportArgumentType]

    plt.colorbar(c, label="Avg Euclidean Distance")
    plt.tight_layout()
    plt.show()

def plot_average_variable(activations: np.ndarray, variable: str, kn: KohonenNet, labels: list[str], europe_df: pd.DataFrame):
    k = kn.k
    avg_variable_values = np.zeros((k, k))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    label_dict = [[{} for _ in range(k)] for _ in range(k)]

    for i in range(k):
        for j in range(k):
            act = activations[i][j]
            total = act.get("total_activations", 0)
            if total > 0:
                sum_val = 0
                for key, count in act.items():
                    if key != "total_activations" and count > 0:
                        idx = label_to_idx[key]
                        sum_val += europe_df.at[europe_df.index[idx], variable] * count
                        label_dict[i][j][key] = count
                avg_variable_values[i, j] = sum_val / total

    cmap = plt.get_cmap('magma')
    norm = plt.Normalize(vmin=np.min(avg_variable_values), vmax=np.max(avg_variable_values)) # pyright: ignore[reportPrivateImportUsage]

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(avg_variable_values, annot=False, cmap=cmap, cbar_kws={
        "label": f"Average {variable} for each neuron"
    })

    for i in range(k):
        for j in range(k):
            labels_in_cell = label_dict[i][j]
            if labels_in_cell:
                sorted_labels = sorted(labels_in_cell.items(), key=lambda x: -x[1])
                text = "\n".join(f"{label}: {cnt}" for label, cnt in sorted_labels)

                rgba = cmap(norm(avg_variable_values[i, j]))
                r, g, b, _ = rgba
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'black' if luminance > 0.5 else 'white'

                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=text_color, fontsize=8)

    plt.title(f"{variable} by neuron")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.tight_layout()
    plt.show()
