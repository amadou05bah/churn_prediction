# Prédiction de l'attrition client (churn)

Un opérateur télécom perd environ un client sur quatre. L'objectif de ce projet est de repérer **à l'avance** les clients qui risquent de résilier, pour que l'entreprise puisse les cibler avec une offre de fidélisation.

Le projet couvre toute la chaîne : nettoyage, analyse exploratoire, feature engineering, pipeline Scikit-learn, comparaison et optimisation de modèles, évaluation et interprétation.

**Données :** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM), 7 043 clients décrits par leur contrat, leurs services souscrits, leur facturation et leur ancienneté.

## Résultats clés

Le modèle retenu, une **forêt aléatoire optimisée**, détecte **79 % des clients qui partent** sur le jeu de test, avec une **AUC ROC de 0,84**.

| Modèle (jeu de test) | AUC ROC | Rappel | Précision | F1 |
|---|---|---|---|---|
| Forêt aléatoire optimisée | 0,84 | 0,79 | 0,53 | 0,64 |
| Régression logistique optimisée | 0,84 | 0,79 | 0,50 | 0,61 |
<!-- Vérifie ces valeurs avec ta propre exécution du notebook -->

Concrètement, sur 10 clients signalés comme « à risque », environ 5 partiraient réellement. Pour une campagne de fidélisation, c'est un compromis raisonnable : contacter un client fidèle par erreur coûte bien moins cher que d'en perdre un.

![Matrices de confusion et courbes ROC](images/evaluation.png)
<!-- Exporte le graphique depuis le notebook (clic droit > Enregistrer l'image) -->

## Ce que disent les données

Le **type de contrat** est de loin le facteur le plus déterminant : 43 % des clients au mois partent, contre 3 % des clients engagés sur deux ans. Viennent ensuite l'**ancienneté** (près de la moitié des départs ont lieu la première année) et le **type de connexion** : les clients fibre partent deux fois plus que les clients DSL.

![Taux de churn selon les variables clés](images/exploration.png)

**Recommandation :** cibler en priorité les nouveaux clients en contrat mensuel, par exemple avec une offre incitant à passer à un engagement d'un an.

## Démarche

**Choix de la métrique.** Avec 26 % de churners, un modèle qui prédirait « personne ne part » aurait déjà 74 % d'exactitude. J'ai donc utilisé l'**AUC ROC** pour comparer les modèles, et le **rappel** pour juger leur utilité métier.

**Nettoyage.** `TotalCharges` était stocké en texte, avec 11 valeurs vides. Elles correspondaient toutes à des clients inscrits depuis 0 mois, jamais encore facturés : je les ai remplacées par 0 plutôt que de supprimer ces clients.

**Pipeline.** Le prétraitement (standardisation des variables numériques, encodage one-hot des catégorielles) est intégré dans un `Pipeline` avec `ColumnTransformer`. Il est ainsi réappris dans chaque pli de validation croisée, ce qui évite toute fuite de données, et le modèle sauvegardé peut prédire directement sur des données brutes.

**Comparaison des modèles.** Régression logistique, arbre de décision, forêt aléatoire et SVM, évalués par validation croisée stratifiée à 5 plis, tous avec des classes pondérées :

| Modèle (validation croisée) | AUC ROC | Rappel | Précision | F1 |
|---|---|---|---|---|
| Régression logistique | 0,846 | 0,798 | 0,518 | 0,628 |
| Arbre de décision | 0,828 | 0,777 | 0,509 | 0,615 |
| Forêt aléatoire | 0,825 | 0,467 | 0,639 | 0,539 |
| SVM | 0,825 | 0,779 | 0,523 | 0,626 |

**Optimisation.** Les deux meilleurs candidats ont été optimisés par `GridSearchCV`. La forêt aléatoire, qui sur-apprenait avec ses paramètres par défaut (rappel de 47 %), rejoint la régression logistique une fois sa profondeur limitée. Le modèle final a été choisi sur la validation croisée, et non sur le jeu de test, pour que le score de test reste une estimation honnête.

**Interprétation.** L'importance des variables est mesurée par permutation sur le jeu de test.

## Ce que j'ai appris

<!-- Écris cette partie toi-même : une difficulté rencontrée, ce qui t'a surpris, ce que tu ferais différemment. Par exemple : -->

- [ex. Ma première version affichait un bon score d'exactitude, mais ne détectait que la moitié des départs. C'est en regardant la matrice de confusion que j'ai compris l'importance de pondérer les classes.]
- [ex. Les variables que j'avais créées (nombre de services, tranches d'ancienneté) n'ont presque rien apporté au modèle : leur information était déjà contenue dans les variables d'origine.]

## Pistes d'amélioration

Ajuster le seuil de décision selon le coût réel d'une action de fidélisation, tester des modèles de boosting (XGBoost, LightGBM) et étudier plus en détail le cas des clients fibre.

## Lancer le projet

Le plus simple est d'ouvrir `churn_prediction.ipynb` dans [Google Colab](https://colab.research.google.com/) et de tout exécuter : le dataset est téléchargé automatiquement via `kagglehub`.

En local :

```bash
git clone https://github.com/amadou05bah/churn-prediction.git
cd churn-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

**Outils :** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib

---

Amadou Bah, étudiant ingénieur Data & IA à l'INSA Rennes · [LinkedIn](https://www.linkedin.com/in/amadou-bah)
