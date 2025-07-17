import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import warnings
warnings.filterwarnings("ignore")

class GustafsonKessel:
    def __init__(self, n_clusters, m=2, max_iter=100, error=1e-5, seed=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        np.random.seed(seed)

    def fit_predict(self, X):
        N, d = X.shape
        U = np.random.dirichlet(np.ones(self.n_clusters), size=N)
        V = np.zeros((self.n_clusters, d))
        A = np.array([np.eye(d) for _ in range(self.n_clusters)])
        for iteration in range(self.max_iter):
            U_old = U.copy()
            for k in range(self.n_clusters):
                um = U[:,k] ** self.m
                V[k] = np.sum(um[:,None] * X, axis=0) / np.sum(um)
            for k in range(self.n_clusters):
                um = U[:,k] ** self.m
                diff = X - V[k]
                A[k] = np.dot((um[:,None] * diff).T, diff) / np.sum(um)
                A[k] += 1e-6 * np.eye(d)
            for i in range(N):
                for k in range(self.n_clusters):
                    diff = X[i] - V[k]
                    dist = np.dot(np.dot(diff, np.linalg.inv(A[k])), diff)
                    U[i,k] = 1.0 / np.sum([
                        (dist / np.dot(np.dot(X[i] - V[j], np.linalg.inv(A[j])), X[i] - V[j])) ** (1/(self.m-1))
                        for j in range(self.n_clusters)
                    ])
            if np.linalg.norm(U - U_old) < self.error:
                break
        labels = np.argmax(U, axis=1)
        return labels

def run_kmeans(X, n_clusters, seed):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(X)
    return labels

def run_fuzzy_cmeans(X, n_clusters, seed):
    fcm = FCM(n_clusters=n_clusters, random_state=seed)
    fcm.fit(X)
    labels = fcm.predict(X)
    return labels

def run_gustafson_kessel(X, n_clusters, seed):
    gk = GustafsonKessel(n_clusters=n_clusters, seed=seed)
    labels = gk.fit_predict(X)
    return labels

def run_mountain_clustering(X, n_clusters, grid_size=10, sigma=1.0, n_components=2, seed=42):
    np.random.seed(seed)
    if X.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    X_norm = (X_reduced - X_reduced.min(axis=0)) / (X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-8)
    grid = [np.linspace(0,1,grid_size) for _ in range(X_norm.shape[1])]
    mesh = np.meshgrid(*grid)
    grid_points = np.vstack([m.flatten() for m in mesh]).T
    M = np.zeros(len(grid_points))
    for i, gp in enumerate(grid_points):
        M[i] = np.sum(np.exp(-np.sum((X_norm - gp)**2, axis=1) / (2*sigma**2)))
    centers = []
    for _ in range(n_clusters):
        idx = np.argmax(M)
        centers.append(grid_points[idx])
        M -= M[idx] * np.exp(-np.sum((grid_points - grid_points[idx])**2, axis=1) / (2*sigma**2))
    labels = np.argmin(cdist(X_norm, np.array(centers)), axis=1)
    return labels

def best_map(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    mapping = dict(zip(ind[0], ind[1]))
    y_pred_mapped = np.array([mapping[yi] for yi in y_pred])
    return y_pred_mapped

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, kappa, cm

def run_experiment(X, y, n_clusters, dataset_name):
    results = {'kmeans':[], 'fcm':[], 'gk':[], 'mountain':[]}
    for seed in range(1, 31):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        # KMeans
        labels = run_kmeans(X_train, n_clusters, seed)
        labels_test = run_kmeans(X_test, n_clusters, seed)
        mapped = best_map(y_train, labels)
        mapped_test = best_map(y_test, labels_test)
        acc, f1, kappa, cm = evaluate(y_test, mapped_test)
        results['kmeans'].append([acc, f1, kappa])
        # Fuzzy C-Means
        labels = run_fuzzy_cmeans(X_train, n_clusters, seed)
        labels_test = run_fuzzy_cmeans(X_test, n_clusters, seed)
        mapped = best_map(y_train, labels)
        mapped_test = best_map(y_test, labels_test)
        acc, f1, kappa, cm = evaluate(y_test, mapped_test)
        results['fcm'].append([acc, f1, kappa])
        # Gustafson-Kessel
        labels = run_gustafson_kessel(X_train, n_clusters, seed)
        labels_test = run_gustafson_kessel(X_test, n_clusters, seed)
        mapped = best_map(y_train, labels)
        mapped_test = best_map(y_test, labels_test)
        acc, f1, kappa, cm = evaluate(y_test, mapped_test)
        results['gk'].append([acc, f1, kappa])
        # Mountain Clustering
        labels = run_mountain_clustering(X_train, n_clusters, seed=seed)
        labels_test = run_mountain_clustering(X_test, n_clusters, seed=seed)
        mapped = best_map(y_train, labels)
        mapped_test = best_map(y_test, labels_test)
        acc, f1, kappa, cm = evaluate(y_test, mapped_test)
        results['mountain'].append([acc, f1, kappa])
    print(f"\n=== {dataset_name} ===")
    for alg in results:
        arr = np.array(results[alg])
        print(f"{alg}: ACC={arr[:,0].mean():.3f}±{arr[:,0].std():.3f} | F1={arr[:,1].mean():.3f}±{arr[:,1].std():.3f} | Kappa={arr[:,2].mean():.3f}±{arr[:,2].std():.3f}")
    # Friedman test
    accs = [np.array(results[alg])[:,0] for alg in results]
    stat, p = friedmanchisquare(*accs)
    print(f"Friedman test ACC: stat={stat:.3f}, p={p:.3g}")
    if p < 0.05:
        nemenyi = sp.posthoc_nemenyi_friedman(np.array(accs).T)
        print("Nemenyi post-hoc (ACC):\n", nemenyi)
    return results

if __name__ == "__main__":
    # Winequality from CSV
    df_wine = pd.read_csv('datasets/winequality-red.csv', sep=',')
    X_wine = df_wine.iloc[:, :-1].values
    y_wine = df_wine.iloc[:, -1].values
    scaler = StandardScaler()
    X_wine_scaled = scaler.fit_transform(X_wine)
    n_clusters_wine = len(np.unique(y_wine))
    run_experiment(X_wine_scaled, y_wine, n_clusters_wine, "Winequality")

    # Heart Dataset
    df_heart = pd.read_csv('datasets/heart.csv')
    X_heart = df_heart.drop('target', axis=1).values
    y_heart = df_heart['target'].values
    scaler = StandardScaler()
    X_heart_scaled = scaler.fit_transform(X_heart)
    n_clusters_heart = len(np.unique(y_heart))
    run_experiment(X_heart_scaled, y_heart, n_clusters_heart, "Heart")

