import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import cluster
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from collections import Counter

project = pd.read_csv('WineQT.csv')
print(project)
print(project.head())
print(project.describe())
print(project.tail())
print(project.columns.tolist())
X = project[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol', 'Id']]
y = project['quality']
print(X.dtypes) 
print(y.dtypes)
duplicati = project.duplicated().sum()                                      
print(duplicati)
mancanti = project.isnull().sum()
print(mancanti) 
project['Alcohol Category'] = pd.cut(project['alcohol'],
                                     bins=[0, 10, 11.5, 15], 
                                     labels=['Basso', 'Medio', 'Alto'])
plt.figure(figsize=(8, 6))
sns.barplot(x='Alcohol Category', y='quality', data=project, palette='coolwarm')
plt.title('Qualità media del vino per fascia di alcol')
plt.xlabel('Categoria di Alcol')
plt.ylabel('Qualità media')
plt.show()
features = project.drop(["quality", "Id"], axis=1)
features = features.select_dtypes(include=['int64', 'float64'])
target = project['quality']
sc = StandardScaler()
scaled_data = sc.fit_transform(features)
data_std = pd.DataFrame(scaled_data, columns=features.columns)
data_std['quality'] = target 
cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol']
grouped_std = data_std.groupby('quality')[cols].mean()
grouped_std.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title('Valori medi (standardizzati) per ciascun livello di qualità del vino')
plt.xlabel('Qualità del vino')
plt.ylabel('Valore standardizzato medio')
plt.xticks(rotation=0)
plt.legend(title='Variabile')
plt.tight_layout()
plt.show()
corr = data_std.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": 0.8}, linewidths=0.5)
plt.title('Matrice di correlazione (dati standardizzati)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
data_std['quality_simplified'] = pd.cut(data_std['quality'],
                                  bins=[2, 5, 6, 9],
                                  labels=['Bassa', 'Media', 'Alta'])
X = data_std.drop(columns=['quality', 'quality_simplified'])
y = data_std['quality_simplified'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=4, min_samples_leaf=3, random_state=0)
clf.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['Bassa', 'Media', 'Alta'], filled=True, rounded=True)
plt.title("Decision Tree (con dati standardizzati)")
plt.show()
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport di classificazione:\n", classification_report(y_test, y_pred, target_names=['Bassa', 'Media', 'Alta']))
confusion_matrix(y_test, y_pred)
ordered_labels = ['Bassa', 'Media','Alta']
cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ordered_labels)
disp.plot()
plt.show()
models = {
    "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=6), n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=42)
}
trainAcc = []
testAcc = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    trainAcc.append(accuracy_score(y_train, y_train_pred))
    testAcc.append(accuracy_score(y_test, y_test_pred))
    print(f"\n=== {name} ===")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Bassa', 'Media', 'Alta']))
methods = list(models.keys())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.bar(np.arange(len(methods)), trainAcc, color='skyblue')
ax1.set_title("Train Accuracy")
ax1.set_xticks(np.arange(len(methods)))
ax1.set_xticklabels(methods)
ax1.set_ylim(0.4, 1)
ax2.bar(np.arange(len(methods)), testAcc, color='lightgreen')
ax2.set_title("Test Accuracy")
ax2.set_xticks(np.arange(len(methods)))
ax2.set_xticklabels(methods)
ax2.set_ylim(0.4, 1)
plt.suptitle("Confronto tra Ensemble Methods sul dataset del vino")
plt.show()
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Media accuratezza cross-validation:", cv_scores.mean())
y_train = y_train.astype(str)
y_test = y_test.astype(str)
k = 35
knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=1)
knn.fit(X_train, y_train)
y_pred_knn_train = knn.predict(X_train)
y_pred_knn_test = knn.predict(X_test)
print("=== KNN ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_knn_train)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_knn_test)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_knn_test, target_names=['Bassa', 'Media', 'Alta']))
y_pred_tree_train = clf.predict(X_train)
y_pred_tree_test = clf.predict(X_test)
print("\n=== Decision Tree ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_tree_train)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_tree_test)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_tree_test, target_names=['Bassa', 'Media', 'Alta']))
cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
scores = cross_val_score(knn, X, y, cv=cross_validation, scoring='accuracy')
print(scores)
print(f"Mean Accuracy: {scores.mean():.2f}")
confusion_matrix(y_test, y_pred_knn_test)
ordered_labels = ['Bassa', 'Media','Alta']
cm2 = confusion_matrix(y_test, y_pred_knn_test, labels=ordered_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ordered_labels)
disp.plot()
plt.show()
maxdepths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))
for i, depth in enumerate(maxdepths):
    y_pred_tree_train = clf.predict(X_train)
    y_pred_tree_test = clf.predict(X_test)

    trainAcc[i] = accuracy_score(y_train, y_pred_tree_train)
    testAcc[i] = accuracy_score(y_test, y_pred_tree_test)
diff = np.abs(trainAcc - testAcc)
score = testAcc - diff
best_index = np.argmax(score)
best_depth = maxdepths[best_index]
print(f"✅ Profondità ottimale: {best_depth}")
print(f"   Train Accuracy: {trainAcc[best_index]:.3f}")
print(f"   Test Accuracy : {testAcc[best_index]:.3f}")
plt.figure(figsize=(10,6))
plt.plot(maxdepths, trainAcc, 'ro-', label='Training Accuracy')
plt.plot(maxdepths, testAcc, 'bv--', label='Test Accuracy')
plt.axvline(x=best_depth, color='g', linestyle='--', label=f'Miglior profondità = {best_depth}')
plt.scatter(best_depth, testAcc[best_index], c='green', s=100, edgecolors='k')
plt.xlabel('Profondità massima dell\'albero (max_depth)')
plt.ylabel('Accuracy')
plt.title('Overfitting/Underfitting nel Decision Tree')
plt.legend()
plt.grid(True)
plt.show()
k_values = list(range(1, 41))
trainAcc = np.zeros(len(k_values))
testAcc = np.zeros(len(k_values))
for i, k in enumerate(k_values):
    y_train_knn_pred = knn.predict(X_train)
    y_test_knn_pred = knn.predict(X_test)

    trainAcc[i] = accuracy_score(y_train, y_train_knn_pred)
    testAcc[i] = accuracy_score(y_test, y_test_knn_pred)
plt.figure(figsize=(10,6))
plt.plot(k_values, trainAcc, 'ro-', label='Training Accuracy')
plt.plot(k_values, testAcc, 'bv--', label='Test Accuracy')
plt.xlabel('Numero di vicini (k)')
plt.ylabel('Accuracy')
plt.title('Overfitting/Underfitting nel KNN')
plt.legend()
plt.grid(True)
plt.show()
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_test)
print("=== Gaussian Naive Bayes ===")
print(f"Train Accuracy: {accuracy_score(y_train, gnb.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_nb)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_nb, target_names=['Bassa', 'Media', 'Alta']))
svm_clf = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("=== Support Vector Machine (RBF) ===")
print(f"Train Accuracy: {accuracy_score(y_train, svm_clf.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print("Report di classificazione:\n", classification_report(y_test, y_pred_svm, target_names=['Bassa', 'Media', 'Alta']))
cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
scores = cross_val_score(svm_clf, X, y, cv=cross_validation, scoring='accuracy')
print(scores)
print(f"Mean Accuracy: {scores.mean():.2f}")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = svm_pca.predict(X_test_pca)
print("Accuracy su test (PCA + SVM):", accuracy_score(y_test_pca, y_pred_pca))
le = LabelEncoder()
y_encoded = le.fit_transform(y)
h = .02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = le.transform(Z)  # Converti da stringhe a numeri
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=y_encoded, cmap=plt.cm.coolwarm, edgecolors='k', s=30)
plt.title("Confine decisionale SVM con PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Classi", labels=le.classes_)
plt.tight_layout()
plt.show()
y_pred_full = svm_pca.predict(X_pca)
error_mask = y != y_pred_full
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_pred_encoded = label_encoder.transform(y_pred_full)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[~error_mask, 0], X_pca[~error_mask, 1],
            c=y_encoded[~error_mask], cmap=plt.cm.coolwarm,
            edgecolors='k', s=30, alpha=0.7, label='Corretti')
plt.scatter(X_pca[error_mask, 0], X_pca[error_mask, 1],
            c=y_encoded[error_mask], cmap=plt.cm.coolwarm,
            marker='x', s=50, label='Errori')
plt.title("Errori di classificazione del modello SVM su PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
y_bin = label_binarize(y, classes=['Bassa', 'Media', 'Alta'])
n_classes = y_bin.shape[1]
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)
ovs = OneVsRestClassifier(RandomForestClassifier(random_state=42))
ovs.fit(X_train, y_train_bin)
y_score = ovs.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(8, 6))
colors = ['darkorange', 'green', 'blue']
class_names = ['Bassa', 'Media', 'Alta']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, color=colors[i],
             label=f'Classe {class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Multiclasse')
plt.legend(loc="lower right")
plt.grid()
plt.show()
k_means = cluster.KMeans(n_clusters=3, max_iter=300, random_state=1)
k_means.fit(X)
labels = k_means.labels_
X_clustered = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
X_clustered['Cluster ID'] = labels
centroids = pd.DataFrame(k_means.cluster_centers_, columns=X_clustered.columns[:-1])
SSE = []
numClusters = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for k in numClusters:
    kmeans_test = cluster.KMeans(n_clusters=k, random_state=1)
    kmeans_test.fit(X)
    SSE.append(kmeans_test.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(numClusters, SSE, marker='o')
plt.xlabel('Numero di Cluster')
plt.ylabel('SSE')
plt.title('Metodo del Gomito per KMeans')
plt.grid(True)
plt.show()
data_std = data_std.drop(["quality_simplified"], axis=1)
sample_labels = data_std.index.astype(str).tolist()
Z_average = hierarchy.linkage(X, method='average')
plt.figure(figsize=(12, 8))
hierarchy.dendrogram(Z_average, labels=sample_labels, orientation='right', leaf_font_size=7)
plt.title("Hierarchical Clustering - Average Link")
plt.xlabel("Distanza")
plt.ylabel("Campioni (indice)")
plt.tight_layout()
plt.show()
Z_average = hierarchy.linkage(data_std, method='average')
cluster_labels = fcluster(Z_average, t=3, criterion='maxclust')  #voglio ottenere aL massimo 3 clust
distance_matrix = euclidean_distances(data_std)
sorted_indices = np.argsort(cluster_labels)
sorted_distances = distance_matrix[sorted_indices, :][:, sorted_indices]
plt.figure(figsize=(10, 8))
sns.heatmap(sorted_distances, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Validation Using Sorted Similarity Matrix (Hierarchical Clustering - Average Link)")
plt.xlabel("Campioni ordinati per cluster")
plt.ylabel("Campioni ordinati per cluster")
plt.tight_layout()
plt.show()
project = pd.read_csv('WineQT.csv')
wine_ids = project['Id']
features_to_scale = project.drop(columns=['Id'])
scaled = sc.fit_transform(features_to_scale)
scaled_df = pd.DataFrame(scaled, columns=features_to_scale.columns, index=project.index)
scaled_df['Id'] = wine_ids
item_profiles_std = {
    scaled_df.iloc[i]['Id']: scaled[i]
    for i in range(len(scaled_df))
}
np.random.seed(42)
num_users = 10
wine_ids = scaled_df['Id'].unique()
ratings_data = []
for user_id in range(1, num_users + 1):
    rated_wines = np.random.choice(wine_ids, size=np.random.randint(30, 100), replace=False)
    for wine_id in rated_wines:
        base_quality = scaled_df.loc[scaled_df['Id'] == wine_id, 'quality'].values[0]
        simulated_rating = np.clip(np.random.normal(loc=base_quality, scale=0.5), 3, 8)
        ratings_data.append([user_id, wine_id, round(simulated_rating)])
ratings = pd.DataFrame(ratings_data, columns=["userId", "wineId", "rating"])
user_profiles_std = {}
for u in ratings['userId'].unique():
    dfU = ratings[ratings['userId'] == u]
    user_movies = dfU["wineId"].to_numpy()
    user_rating = dfU["rating"].to_numpy()
    user_item_profiles = np.array([item_profiles_std[i] for i in user_movies])
    user_item_profiles_rated = np.array([uip * ur for uip, ur in zip(user_item_profiles, user_rating)])
    user_profile = user_item_profiles_rated.sum(axis=0) / len(user_item_profiles_rated)
    user_profiles_std[u] = user_profile
user_profiles_std[1]
u = 1
user_profile = user_profiles_std[u]
rated_wines = set(ratings[ratings['userId'] == u]['wineId'])
all_wines = set(scaled_df['Id'])
unrated_wines = list(all_wines - rated_wines)
recommendations = []
for wine_id in unrated_wines:
    wine_profile = item_profiles_std[wine_id]
    similarity = cosine_similarity([user_profile], [wine_profile])[0][0]
    recommendations.append((wine_id, similarity))
top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
recommended_wines = project[project['Id'].isin([wine[0] for wine in top_recommendations])].copy()
recommended_wines["similarity"] = [sim for _, sim in top_recommendations]
print("Top 5 vini raccomandati per l'utente", u)
display(recommended_wines)
X_scaled = scaled_df.drop("Id", axis=1).values
similarity_matrix = cosine_similarity(X_scaled)
G = nx.DiGraph()
wine_ids = scaled_df['Id'].tolist()
G.add_nodes_from(wine_ids)
threshold = 0.9
for i, id_i in enumerate(wine_ids):
    for j, id_j in enumerate(wine_ids):
        if i != j and similarity_matrix[i, j] > threshold:
            G.add_edge(id_i, id_j, weight=similarity_matrix[i, j])
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')
top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
top_vini = project[project['Id'].isin([wine_id for wine_id, _ in top_pagerank])].copy()
top_vini["PageRank"] = [score for _, score in top_pagerank]
print("Top 5 vini secondo PageRank:")
display(top_vini)
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.15, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=30, node_color="lightblue")
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color="gray", width=0.5)
plt.title("Grafo vino-vino basato su similarità chimica (cosine > 0.9)")
plt.axis("off")
plt.show()
pos = nx.spring_layout(G, k=0.15, seed=42)
pr_values = np.array(list(pagerank_scores.values()))
pr_min, pr_max = pr_values.min(), pr_values.max()
plt.figure(figsize=(14, 12))
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=30,
    node_color=pr_values,
    cmap=plt.cm.plasma,
    vmin=pr_min, vmax=pr_max  
)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=6, edge_color="gray", width=0.3)
plt.colorbar(nodes, label="PageRank (normalizzato)")
plt.title("Grafo vino-vino con nodi colorati per PageRank")
plt.axis("off")
plt.show()
hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')
centrality_df = pd.DataFrame({
    "Id": list(G.nodes),
    "HITS Hub": [hubs[n] for n in G.nodes],
    "HITS Authority": [authorities[n] for n in G.nodes],
    "PageRank": [pagerank_scores[n] for n in G.nodes]
})
print("Metriche di Link Analysis per ciascun vino:")
display(centrality_df.sort_values(by="PageRank", ascending=False).head(10))
plt.figure(figsize=(8, 6))
sns.scatterplot(data=centrality_df, x="PageRank", y="HITS Authority", s=40, alpha=0.7, color="orange")
plt.title("Confronto: PageRank vs HITS Authority")
plt.xlabel("PageRank")
plt.ylabel("HITS Authority")
plt.grid(True)
plt.tight_layout()
plt.show()
device = "cuda" if torch.cuda.is_available() else "cpu"
project = pd.read_csv("WineQT.csv")
X = project.drop("quality", axis=1)
y = project["quality"]
y_cat = y.copy().to_numpy()
y_cat[y_cat <= 4] = 0
y_cat[(y_cat == 5) | (y_cat == 6)] = 1
y_cat[y_cat >= 7] = 2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.3, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
class MLP(nn.Module):
    def __init__(self, input_size, hidden1=32, hidden2=16, output_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x):
        return self.model(x)
model = MLP(input_size=X_train.shape[1])
counts = Counter(y_train)
print(counts)
total = sum(counts.values())
weights = [total / counts[i] for i in range(3)]
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001)
y_logits = model(X_test_tensor.to(device))
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])
loss_train=[]
acc_train=[]
epochs=2000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred_labels = torch.argmax(preds, dim=1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)
    epoch_acc = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    loss_train.append(epoch_loss)
    acc_train.append(epoch_acc)
    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = predictions.argmax(dim=1).numpy()
    true_classes = y_test_tensor.numpy()
print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes))
print(f"Test Accuracy: {accuracy_score(true_classes, predicted_classes) * 100:.2f}%")
plt.subplot(1, 2, 1)
plt.plot(loss_train, label='Train Loss')
plt.title('Loss per epoca')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc_train, label='Train Accuracy', color='green')
plt.title('Accuracy per epoca')
plt.xlabel('Epoca')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()