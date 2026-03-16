from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
import random
from kneed import KneeLocator
import matplotlib.pyplot as plt

# Important: this is the number of clusters used in the clustering scripts
global_k = 5

# Which summary statistic to use when aggregating per-image embeddings to per-LSOA.
# All scripts that aggregate embeddings should import and use this value.
# Options: 'mean', 'median', 'max'
embedding_statistic = 'median'

# Random seed for reproducibility (used across all scripts)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Embedding aggregation helpers
# Used by scripts 6, 7, 8 (and anywhere else that needs to reduce a Series
# of embedding vectors to a single summary vector).
# ---------------------------------------------------------------------------

def mean_embed(series):
    """Element-wise mean of a Series of embedding vectors."""
    return np.mean(np.stack(series.values), axis=0)

def median_embed(series):
    """Element-wise median of a Series of embedding vectors."""
    return np.median(np.stack(series.values), axis=0)

def max_embed(series):
    """Element-wise max of a Series of embedding vectors."""
    return np.max(np.stack(series.values), axis=0)

# Lookup dict so callers can do:  agg_funcs[embedding_statistic](series)
agg_funcs = {"mean": mean_embed, "median": median_embed, "max": max_embed}

def clustering_stability(all_embeddings, k, n_runs=10):
    """
    Run k-means multiple times with different random initialisations
    and compute pairwise Adjusted Rand Index (ARI).
    """
    labels_list = []

    for seed in range(n_runs):
        kmeans = KMeans(
            n_clusters=k,
            n_init=1,          # important: single init per run
            random_state=seed
        )
        labels = kmeans.fit_predict(all_embeddings)
        labels_list.append(labels)

    # Compute pairwise ARI
    ari_scores = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            ari_scores.append(ari)

    return np.mean(ari_scores), np.std(ari_scores), ari_scores

def find_optimal_k_elbow(all_embeddings):
    """
    Uses the elbow method (inertia) and automatically detects
    the elbow using the KneeLocator algorithm.
    """
    n_samples = len(all_embeddings)
    max_k = min(20, n_samples - 1)
    k_values = range(2, max_k + 1)

    inertias = []

    #print("Computing inertias...")
    for k in k_values:
        #print(f"k = {k}")
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(all_embeddings)
        inertias.append(kmeans.inertia_)

    # Use Kneedle to find elbow point
    kn = KneeLocator(
        k_values,
        inertias,
        curve='convex',
        direction='decreasing')

    optimal_k = kn.knee

    print(f"\nDetected elbow at k = {optimal_k}")
    return optimal_k, k_values, inertias

def find_optimal_k_silhoutte_fast(all_embeddings, max_k=20, sample_size=5000):
    n_samples = all_embeddings.shape[0]
    max_k = min(max_k, n_samples - 1)
    k_values = range(2, max_k + 1)
    
    inertias = []
    sil_scores = []
    rng = check_random_state(RANDOM_STATE)
    
    for k in k_values:
        #print(f"Testing k={k}")s
        
        # MiniBatchKMeans is much faster for large datasets
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=1024)
        labels = kmeans.fit_predict(all_embeddings)
        inertias.append(kmeans.inertia_)
        
        # Silhouette on a random subset to reduce computation
        if n_samples > sample_size:
            idx = rng.choice(n_samples, sample_size, replace=False)
            sil = silhouette_score(all_embeddings[idx], labels[idx])
        else:
            sil = silhouette_score(all_embeddings, labels)
        
        sil_scores.append(sil)
        #print(f"Silhouette score: {sil:.4f}")
    
    best_k = k_values[np.argmax(sil_scores)]
    return best_k, inertias, sil_scores



# def plot_elbow(k_values, inertias, optimal_k):
#     plt.figure(figsize=(5,3))
#     plt.plot(k_values, inertias, marker='o')
#     plt.axvline(optimal_k, linestyle='--')
#     plt.title("Elbow Method")
#     plt.xlabel("Number of clusters (k)")
#     plt.ylabel("Inertia")
#     plt.show()


# def find_optimal_k(all_embeddings):
#     # --- Range of cluster counts to test ---
#     n_samples = len(all_embeddings)
#     max_k = min(20, n_samples - 1)
#     k_values = range(2, max_k + 1)

#     inertias = []
#     sil_scores = []

#     for k in k_values:
#         print(k)
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         labels = kmeans.fit_predict(all_embeddings)

#         inertias.append(kmeans.inertia_)
#         sil_scores.append(silhouette_score(all_embeddings, labels))
#         #print(silhouette_score(all_embeddings, labels))
#     best_k = k_values[np.argmax(sil_scores)]
#     return best_k


# def plot_example_images_per_cluster(df, k, column, n_show=5):
#     fixed_rows, fixed_cols = int(n_show/6), 6  # always 6 spaces total

#     for c in np.unique(df[column]):
#         cluster_imgs = df.loc[df[column] == c, 'image_files']
#         if len(cluster_imgs) == 0:
#             continue

#         # Sample up to n_show images
#         sample_imgs = random.sample(list(cluster_imgs), min(n_show, len(cluster_imgs)))

#         fig, axes = plt.subplots(fixed_rows, fixed_cols, figsize=(fixed_cols * 3, fixed_rows * 3))
#         axes = np.array(axes).reshape(-1)  # flatten axes for easy indexing

#         for ax, img_path in zip(axes, sample_imgs):
            
#             try:
#                 #adj_path = img_path.replace("../data/airbnb-manchester/street_images/", "../../data/embeddings/saliency_heatmaps_new/").replace("_new/point", "_new/attn_point")
#                 #adj_path = adj_path.replace(".jpg", ".jpg.png")
#                 adj_path = img_path.replace("airbnb-manchester/", "embeddings/").replace("../", "../../")
#                 img = plt.imread(adj_path)
#                 ax.imshow(img)
#                 ax.axis("off")
#             except:
#                 print("image not found")

#         # Hide any unused subplot spaces
#         for ax in axes[len(sample_imgs):]:
#             ax.axis("off")

#         plt.suptitle(f"Global Cluster {c}, {len(cluster_imgs)} images", fontsize=14, fontweight='bold')
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()
        
# def lsoa_within_between_summary(expanded_gdf):
#     results = []

#     for lsoa, df_lsoa in expanded_gdf.groupby('LSOA11CD'):
#         embeddings = np.stack(df_lsoa['embedding'].values)
#         clusters = df_lsoa['scene_cluster'].values
#         n_total = len(embeddings)

#         if n_total < 2:
#             results.append({
#                 'LSOA11CD': lsoa,
#                 'mean_within': np.nan,
#                 'mean_between': np.nan,
#                 'within_minus_between': np.nan,
#                 'n_images': n_total
#             })
#             continue

#         sim_matrix = cosine_similarity(embeddings)
#         within_sims = []
#         between_sims = []

#         for i, j in combinations(range(n_total), 2):
#             sim = sim_matrix[i, j]
#             if clusters[i] == clusters[j]:
#                 within_sims.append(sim)
#             else:
#                 between_sims.append(sim)

#         mean_within = np.mean(within_sims) if within_sims else np.nan
#         mean_between = np.mean(between_sims) if between_sims else np.nan
#         within_minus_between = mean_within - mean_between if mean_within is not np.nan and mean_between is not np.nan else np.nan

#         results.append({
#             'LSOA11CD': lsoa,
#             'mean_within': mean_within,
#             'mean_between': mean_between,
#             'within_minus_between': within_minus_between,
#             'n_images': n_total})

#     df_results = pd.DataFrame(results)

#     # Check for LSOAs where between > within
#     anomalies = df_results[df_results['within_minus_between'] < 0]
#     n_anomalies = len(anomalies)
#     if n_anomalies > 0:
#         print(f"Warning: {n_anomalies} LSOAs have higher between-cluster similarity than within-cluster similarity.")
#     else:
#         print("All LSOAs have higher within-cluster similarity than between-cluster similarity.")

#     return df_results

# def mean_embedding_per_cluster(expanded_gdf, K, embedding_col):
#     """
#     Compute mean embedding per cluster for each LSOA.
    
#     Parameters:
#     - expanded_gdf: DataFrame with columns 'LSOA11CD', 'embedding', 'scene_cluster'
#     - K: total number of clusters (0 to K-1)
    
#     Returns:
#     - DataFrame with one row per LSOA, columns: 'cluster_0', 'cluster_1', ..., 'cluster_{K-1}'
#       Each cell contains the mean embedding vector for that cluster, or np.nan if no images in cluster.
#     """
#     results = []

#     for lsoa, df_lsoa in expanded_gdf.groupby('LSOA11CD'):
#         row = {'LSOA11CD': lsoa}
#         for k in range(K):
#             cluster_embeddings = df_lsoa.loc[df_lsoa['scene_cluster'] == k, embedding_col].values
#             if len(cluster_embeddings) == 0:
#                 row[f'cluster_{k}'] = np.nan  # no images in this cluster
#             else:
#                 row[f'cluster_{k}'] = np.mean(np.stack(cluster_embeddings), axis=0)
#         results.append(row)

#     return pd.DataFrame(results)

# def weighted_mean_embedding_per_cluster(expanded_gdf, K, embedding_col):
#     """
#     Compute weighted mean embedding per LSOA, where each cluster is weighted by its proportion of images.
    
#     Parameters:
#     - expanded_gdf: DataFrame with columns 'LSOA11CD', 'embedding', 'scene_cluster'
#     - K: total number of clusters (0 to K-1)
#     - embedding_col: column containing 1D embedding arrays
    
#     Returns:
#     - DataFrame with one row per LSOA, columns: 'weighted_embedding'
#       Each row contains a single weighted embedding vector for the LSOA.
#     """
#     results = []

#     for lsoa, df_lsoa in expanded_gdf.groupby('LSOA11CD'):
#         n_total = len(df_lsoa)  # total images in LSOA
#         weighted_sum = None

#         for k in range(K):
#             cluster_embeddings = df_lsoa.loc[df_lsoa['scene_cluster'] == k, embedding_col].values
#             n_cluster = len(cluster_embeddings)

#             if n_cluster == 0:
#                 continue  # weight = 0 for missing clusters

#             cluster_mean = np.mean(np.stack(cluster_embeddings), axis=0)
#             weight = n_cluster / n_total

#             if weighted_sum is None:
#                 weighted_sum = weight * cluster_mean
#             else:
#                 weighted_sum += weight * cluster_mean

#         results.append({'LSOA11CD': lsoa, 'weighted_embedding': weighted_sum})

#     return pd.DataFrame(results)

# def cluster_embeddings_with_proportions(expanded_gdf, K, embedding_col):
#     """
#     Compute weighted mean embedding per cluster and cluster proportions for each LSOA.
    
#     Parameters:
#     - expanded_gdf: DataFrame with columns 'LSOA11CD', 'embedding', 'scene_cluster'
#     - K: total number of clusters (0 to K-1)
#     - embedding_col: column containing 1D embedding arrays
    
#     Returns:
#     - DataFrame with one row per LSOA
#       - Columns 'cluster_{k}_embedding': mean embedding of cluster k (NaN if missing)
#       - Columns 'cluster_{k}_proportion': fraction of images in cluster k
#     """
#     results = []

#     for lsoa, df_lsoa in expanded_gdf.groupby('LSOA11CD'):
#         n_total = len(df_lsoa)
#         row = {'LSOA11CD': lsoa}

#         for k in range(K):
#             cluster_embeddings = df_lsoa.loc[df_lsoa['scene_cluster'] == k, embedding_col].values
#             n_cluster = len(cluster_embeddings)
            
#             # proportion of images in this cluster
#             row[f'cluster_{k}_proportion'] = n_cluster / n_total if n_total > 0 else 0.0

#             if n_cluster == 0:
#                 row[f'cluster_{k}_embedding'] = np.nan  # or np.zeros(embedding_dim) if preferred
#             else:
#                 cluster_mean = np.mean(np.stack(cluster_embeddings), axis=0)
#                 row[f'cluster_{k}_embedding'] = cluster_mean

#         results.append(row)

#     return pd.DataFrame(results)

# def flatten_weighted_cluster_features(df_lsoa_features, K, embedding_dim):
#     """
#     Convert per-cluster embeddings and proportions into a single ML-ready feature vector.
    
#     Parameters:
#     - df_lsoa_features: output of cluster_embeddings_with_proportions()
#     - K: number of clusters
#     - embedding_dim: dimension of each embedding vector
    
#     Returns:
#     - X: DataFrame, one row per LSOA, columns = weighted embeddings + optional proportions
#     """
#     rows = []
#     for _, row in df_lsoa_features.iterrows():
#         features = []

#         for k in range(K):
#             # get cluster proportion
#             prop = row[f'cluster_{k}_proportion']
            
#             # get cluster embedding, fill NaN with zeros
#             emb = row[f'cluster_{k}_embedding']
#             if emb is None or (isinstance(emb, float) and np.isnan(emb)):
#                 emb = np.zeros(embedding_dim)
            
#             # weighted embedding
#             weighted_emb = prop * np.array(emb)
#             features.extend(weighted_emb)
            
#             # optionally, also include proportion explicitly
#             features.append(prop)

#         rows.append(features)

#     X = pd.DataFrame(rows, index=df_lsoa_features['LSOA11CD'])
#     return X