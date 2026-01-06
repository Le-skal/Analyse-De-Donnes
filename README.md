## **1. ENCODAGE DES VARIABLES CATÉGORIELLES**

### **Problème de base**
Les modèles de ML ne comprennent que les chiffres. Donc on doit convertir les catégories (texte) en nombres.

### **One-Hot Encoding (variables nominales)**
**C'est quoi une variable nominale ?** Une catégorie sans ordre logique.

**Exemple avec `person_home_ownership`** :
- Valeurs possibles : RENT, OWN, MORTGAGE
- Pas d'ordre : on ne peut pas dire que OWN > RENT ou MORTGAGE > OWN

**Comment ça marche ?**
On crée une colonne binaire (0 ou 1) pour chaque catégorie :

```
Avant :
| person_home_ownership |
|-----------------------|
| RENT                  |
| OWN                   |
| MORTGAGE              |

Après One-Hot Encoding :
| home_RENT | home_OWN | home_MORTGAGE |
|-----------|----------|---------------|
| 1         | 0        | 0             |
| 0         | 1        | 0             |
| 0         | 0        | 1             |
```

### **Ordinal Encoding (variables ordinales)**
**C'est quoi une variable ordinale ?** Une catégorie avec un ordre logique.

**Exemple avec `loan_grade`** :
- Valeurs possibles : A, B, C, D, E, F, G
- IL Y A un ordre : A est meilleur que B, B meilleur que C, etc.

**Comment ça marche ?**
On remplace chaque catégorie par un nombre qui respecte l'ordre :

```
| loan_grade | → | loan_grade_encoded |
|------------|---|-------------------|
| A          | → | 0                 |
| B          | → | 1                 |
| C          | → | 2                 |
| D          | → | 3                 |
| E          | → | 4                 |
| F          | → | 5                 |
| G          | → | 6                 |
```

**Pourquoi pas One-Hot pour loan_grade ?** Parce qu'on perdrait l'information d'ordre. Le modèle ne saurait pas que A < B < C.

---

## **2. STRATIFIED SPLIT**

### **Le problème**
Tu as un dataset déséquilibré :
- 78% de classe 0 (remboursé)
- 22% de classe 1 (défaut)

### **Split normal (MAUVAIS)**
Si tu fais un split aléatoire simple (80% train / 20% test), tu peux avoir un problème :

```
Dataset complet : 78% classe 0, 22% classe 1

Train (80%) : pourrait avoir 82% classe 0, 18% classe 1  ❌
Test (20%)  : pourrait avoir 65% classe 0, 35% classe 1  ❌
```

Les proportions ne sont pas respectées ! Le modèle s'entraîne sur une distribution différente de celle du test.

### **Stratified Split (BON)**
Il **garantit** que les proportions sont identiques partout :

```
Dataset complet : 78% classe 0, 22% classe 1

Train (80%) : 78% classe 0, 22% classe 1  ✅
Test (20%)  : 78% classe 0, 22% classe 1  ✅
```

**Code Python** :
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # ← Ça c'est le stratified split
    random_state=42
)
```

**Pourquoi c'est important ?**
- Évite le biais d'échantillonnage
- Le modèle voit la même distribution en train et en test
- Les métriques de performance sont plus fiables

---

## **3. LES 3 MODÈLES DE MACHINE LEARNING (VERSION SIMPLE)**

### **📊 RÉGRESSION LOGISTIQUE**

**Analogie simple** : C'est comme tracer une ligne pour séparer les bons et les mauvais emprunteurs.

**Comment ça marche ?**

1. **Étape 1 : Combinaison linéaire**
   ```
   z = β₀ + β₁×(revenu) + β₂×(montant_prêt) + β₃×(âge) + ...
   ```
   - β₀, β₁, β₂... sont des poids appris par le modèle
   - z est un score (peut être négatif ou positif)

2. **Étape 2 : Fonction sigmoïde**
   ```
   Probabilité de défaut = 1 / (1 + e^(-z))
   ```
   - Transforme z en probabilité entre 0 et 1
   - Si z est très négatif → probabilité proche de 0 (bon client)
   - Si z est très positif → probabilité proche de 1 (mauvais client)

3. **Décision finale**
   ```
   Si probabilité > 0.5 → Prédiction : DÉFAUT (classe 1)
   Si probabilité < 0.5 → Prédiction : REMBOURSÉ (classe 0)
   ```

**Paramètres principaux** :
- `C` : Régularisation (plus C est petit, plus le modèle est simple)
- `penalty` : Type de régularisation (L1, L2)
- `class_weight` : Ajuste les poids des classes (important pour déséquilibre)

**Avantages** :
- Simple, rapide
- Interprétable (on voit l'impact de chaque variable)

**Limites** :
- Suppose une relation linéaire
- Peu performant si les relations sont complexes

---

### **🌳 RANDOM FOREST**

**Analogie simple** : C'est comme demander l'avis de 100 experts, puis voter pour la décision finale.

**Comment ça marche ?**

1. **Créer plein d'arbres de décision**
   - Chaque arbre pose des questions : "Le revenu est-il > 50k€ ?", "Le grade est-il < C ?"
   - Chaque arbre est entraîné sur un échantillon aléatoire des données

2. **Exemple d'arbre** :
   ```
                   Revenu > 50k ?
                   /            \
                 OUI            NON
                  |              |
         Grade < C ?      Défaut passé ?
         /        \          /        \
       OUI       NON       OUI       NON
        |         |         |         |
     Classe 0  Classe 1  Classe 1  Classe 0
   ```

3. **Vote final**
   - 100 arbres donnent leur prédiction
   - Si 70 arbres disent "Défaut" et 30 disent "Remboursé" → Prédiction finale : DÉFAUT

**Paramètres principaux** :
- `n_estimators` : Nombre d'arbres (ex: 100, 200)
- `max_depth` : Profondeur max de chaque arbre (limite la complexité)
- `min_samples_split` : Nombre minimum d'échantillons pour diviser un nœud
- `class_weight` : Ajuste les poids des classes

**Avantages** :
- Capture les relations non-linéaires
- Robuste aux outliers
- Donne l'importance des variables

**Limites** :
- Plus lent que la régression logistique
- Moins interprétable (boîte noire)

---

### **🚀 GRADIENT BOOSTING (XGBoost)**

**Analogie simple** : C'est comme un élève qui apprend de ses erreurs, encore et encore.

**Comment ça marche ?**

1. **Arbre 1** : Fait des prédictions (avec plein d'erreurs)
   ```
   Vrai : [0, 1, 0, 1, 0]
   Prédit : [0, 0, 0, 1, 1]
   Erreur : [0, 1, 0, 0, -1]  ← Ces erreurs
   ```

2. **Arbre 2** : Entraîné pour **corriger les erreurs de l'Arbre 1**
   ```
   Il apprend à prédire les erreurs : [0, 1, 0, 0, -1]
   ```

3. **Arbre 3** : Entraîné pour **corriger les erreurs résiduelles**
   
4. **Prédiction finale** = Arbre1 + Arbre2 + Arbre3 + ... (avec des poids)

**Différence avec Random Forest** :
- Random Forest : arbres **indépendants** qui votent
- XGBoost : arbres **séquentiels** qui se corrigent mutuellement

**Paramètres principaux** :
- `n_estimators` : Nombre d'arbres séquentiels
- `learning_rate` : Vitesse d'apprentissage (petit = plus prudent)
- `max_depth` : Profondeur des arbres
- `subsample` : Proportion de données utilisées par arbre
- `colsample_bytree` : Proportion de features utilisées par arbre
- `scale_pos_weight` : Ajuste le poids de la classe positive (défaut)

**Avantages** :
- Meilleure performance sur données tabulaires
- Gère nativement les valeurs manquantes
- Régularisation intégrée

**Limites** :
- Plus lent à entraîner
- Plus de paramètres à régler
- Risque d'overfitting si mal paramétré

---

## **RÉSUMÉ POUR TA PRÉSENTATION**

### **Régression Logistique**
"Modèle linéaire qui calcule un score puis le transforme en probabilité avec une fonction sigmoïde. Simple et interprétable, mais limité aux relations linéaires."

### **Random Forest**
"Ensemble de 100+ arbres de décision entraînés sur des échantillons aléatoires. Chaque arbre vote et la majorité l'emporte. Capture les relations complexes et est robuste aux outliers."

### **XGBoost**
"Construction séquentielle d'arbres où chaque nouvel arbre corrige les erreurs du précédent. Optimise progressivement les prédictions. Le plus performant sur données tabulaires."
