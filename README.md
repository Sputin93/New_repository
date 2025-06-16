# Wine Quality Prediction & Recommendation

Questo progetto analizza un dataset di vini rossi con l’obiettivo di:
- Classificare la qualità del vino usando tecniche di machine learning supervisionato (Decision Tree, KNN, SVM, Naive Bayes, Reti Neurali, Ensemble).
- Esplorare relazioni tra caratteristiche chimiche e qualità percepita.
- Realizzare un sistema di raccomandazione di vini, sia basato su profili utente che tramite grafo di similarità vino-vino.

---

## Dataset

- **Nome**: `WineQT.csv`
- **Origine**: Kaggle
- **Caratteristiche**: 11 variabili chimiche + ID + qualità del vino (punteggio da 3 a 8)
- **Target finale**: Classificazione della qualità in tre classi:
  - `Bassa`: qualità ≤ 4  
  - `Media`: qualità 5 o 6  
  - `Alta`: qualità ≥ 7

---

## Modelli applicati

- **Decision Tree**: analisi della profondità ottimale e visualizzazione dell’albero.
- **K-Nearest Neighbors**: tuning del parametro k e cross-validation.
- **Naive Bayes**: probabilistico, semplice ma efficace su dati indipendenti.
- **SVM (RBF)**: supporto ai dati non linearmente separabili.
- **Ensemble Methods**: Bagging e AdaBoost su base Decision Tree.
- **MLP Neural Network**: classificazione multiclasse con bilanciamento delle classi.

---

## Altre analisi

- **Standardizzazione delle variabili**
- **PCA** per visualizzazione in 2D dei dati
- **Heatmap di correlazione**
- **Clustering gerarchico** con dendrogramma
- **KMeans** e metodo del gomito
- **Sistema di raccomandazione**:
  - User-based: profilo utente → cosine similarity
  - Grapho-based: PageRank e HITS su similarità vino-vino
