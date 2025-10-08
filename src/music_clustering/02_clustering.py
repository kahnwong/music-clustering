import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

os.makedirs("data/output", exist_ok=True)


def prepare_features(df):
    aggregated_features = []
    labels = []
    genres = []

    for idx, row in df.iterrows():
        features = row["features"]

        # Compute statistics across beats
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)

        # Concatenate statistics
        aggregated = np.concatenate([mean_features, std_features])
        aggregated_features.append(aggregated)
        labels.append(f"{row['genre']} - {row['title']}")
        genres.append(row["genre"])

    return np.array(aggregated_features), labels, genres


def perform_kmeans(features, n_clusters=4, random_state=42):
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(features_scaled)

    return kmeans, scaler, features_scaled


def visualize_tsne(features_scaled, cluster_labels, labels, genres, n_clusters):
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(features_scaled) - 1)
    )
    features_2d = tsne.fit_transform(features_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by k-means clusters
    scatter1 = ax1.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=1,
    )
    ax1.set_title(
        f"t-SNE Visualization: K-Means Clusters (k={n_clusters})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("t-SNE Component 1", fontsize=12)
    ax1.set_ylabel("t-SNE Component 2", fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label="Cluster")

    # Add labels to points
    for i, label in enumerate(labels):
        ax1.annotate(
            label,
            (features_2d[i, 0], features_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Plot 2: Color by actual genres
    unique_genres = list(set(genres))
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    genre_indices = [genre_to_idx[genre] for genre in genres]

    ax2.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=genre_indices,
        cmap="tab10",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=1,
    )
    ax2.set_title("t-SNE Visualization: True Genres", fontsize=14, fontweight="bold")
    ax2.set_xlabel("t-SNE Component 1", fontsize=12)
    ax2.set_ylabel("t-SNE Component 2", fontsize=12)

    # Create custom legend for genres
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=plt.cm.tab10(genre_to_idx[genre] / 10),
            label=genre,
            edgecolor="black",
        )
        for genre in unique_genres
    ]
    ax2.legend(handles=legend_elements, loc="best", fontsize=10)

    # Add labels to points
    for i, label in enumerate(labels):
        ax2.annotate(
            label,
            (features_2d[i, 0], features_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(
        "data/output/tsne_clustering_visualization.png", dpi=300, bbox_inches="tight"
    )
    print("Visualization saved to data/input/tsne_clustering_visualization.png")
    plt.show()


if __name__ == "__main__":
    # read data
    df = joblib.load("data/input/beat_features.joblib")

    # prep
    features, labels, genres = prepare_features(df)

    # clustering
    n_clusters = 4
    kmeans, scaler, features_scaled = perform_kmeans(features, n_clusters=n_clusters)

    # get cluster assignments
    cluster_labels = kmeans.labels_

    # display results
    print(f"\nK-Means Clustering Results (k={n_clusters}):")
    print("=" * 60)
    for label, cluster, genre in zip(labels, cluster_labels, genres):
        print(f"{label}: Cluster {cluster}")

    print(f"\nCluster centers shape: {kmeans.cluster_centers_.shape}")
    print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")

    # Visualize with t-SNE
    visualize_tsne(features_scaled, cluster_labels, labels, genres, n_clusters)

    # Save the model
    joblib.dump(kmeans, "data/output/kmeans_model.joblib")
    joblib.dump(scaler, "data/output/scaler.joblib")
    print("\nModel and scaler saved to data/input/")
