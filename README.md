# 🖼️ Reconnaissance de Chiffres Manuscrits avec Deep Learning

> 🎯 *Objectif : Apprendre à un ordinateur à reconnaître des chiffres écrits à la main — comme un enfant apprend à lire les nombres.*

Ce projet contient **deux notebooks Jupyter** qui implémentent deux types de réseaux de neurones pour classifier les chiffres manuscrits du célèbre jeu de données **MNIST** :

- **`01-DNN-MNIST.ipynb`** → Réseau de neurones **dense (DNN)** — simple, efficace, parfait pour débuter.
- **`02-CNN-MNIST.ipynb`** → Réseau de neurones **convolutionnel (CNN)** — plus puissant, conçu pour les images.

---

## 🧠 Pourquoi MNIST ?

Le jeu de données **MNIST** est le “Hello World” du Deep Learning. Il contient :

- ✍️ **60 000 images** d’entraînement (chiffres 0 à 9 écrits à la main)
- ✍️ **10 000 images** de test
- 📏 Images en noir et blanc, 28x28 pixels
- 🎯 Objectif : prédire le chiffre représenté dans chaque image

C’est le terrain d’entraînement idéal pour comprendre les bases de la classification d’images avec les réseaux de neurones.

---

## 👥 Pour qui est ce projet ?

| Public | Ce qu’il y trouvera |
|--------|----------------------|
| 👩‍🎓 **Étudiants / Débutants en IA** | Un tutoriel étape par étape, avec du code simple, des explications claires, et des visualisations pour comprendre comment fonctionne un réseau de neurones. |
| 👨‍🏫 **Enseignants / Formateurs** | Un support pédagogique complet pour enseigner les DNN et CNN, avec évaluation, historique d’entraînement, matrices de confusion… |
| 👩‍💻 **Data Scientists / Développeurs** | Une implémentation propre avec Keras/TensorFlow, facile à modifier, étendre ou comparer. Parfait pour expérimenter. |
| 👔 **Curieux / Non-techniciens** | Des explications simples, des images parlantes, et une démonstration concrète de comment les machines “voient” et “reconnaissent” les chiffres. |

---

## ⚙️ Ce que vous allez apprendre

### ✅ Dans les deux notebooks :
- Charger et normaliser les données MNIST
- Visualiser les images d’entraînement
- Compiler un modèle avec `Adam`, `sparse_categorical_crossentropy`, `accuracy`
- Entraîner le modèle sur 16 époques
- Évaluer les performances (précision, perte)
- Visualiser l’historique d’entraînement
- Afficher les prédictions (bonnes et mauvaises)
- Générer une **matrice de confusion**

### 🧱 01-DNN-MNIST.ipynb — Réseau Dense (Fully Connected)
- Architecture simple :
  - `Flatten()` → transformer l’image 28x28 en vecteur de 784 valeurs
  - 2 couches cachées de 100 neurones avec activation `ReLU`
  - Couche de sortie de 10 neurones avec `softmax` (probabilités pour chaque chiffre)
- Précision attendue : **~97.7%**

### 🧩 02-CNN-MNIST.ipynb — Réseau Convolutionnel
- Architecture plus adaptée aux images :
  - Couches `Conv2D` → détectent les motifs locaux (bords, courbes…)
  - Couches `MaxPooling2D` → réduisent la taille tout en conservant les motifs importants
  - Couches `Dropout` → évitent le surapprentissage
  - Couche `Flatten` + `Dense` → classification finale
- Précision attendue : **> 98.5%** (souvent ~99%)

---

## 📊 Étapes Techniques Réalisées

### 1. 🔢 Préparation des données
- Normalisation des pixels entre 0 et 1
- Pour le CNN : ajout d’une dimension de canal (`reshape(-1,28,28,1)`)

### 2. 🏗️ Construction du modèle
- Utilisation de `keras.Sequential`
- Choix des activations (`relu`, `softmax`)
- Compilation avec `sparse_categorical_crossentropy` (car labels entiers, pas one-hot)

### 3. 🚀 Entraînement
- `batch_size = 512`
- `epochs = 16`
- Validation sur les données de test en direct

### 4. 📈 Évaluation & Visualisation
- Calcul de la précision finale
- Graphique de l’historique (perte et précision sur train/test)
- Affichage des prédictions (vert = bon, rouge = erreur)
- Matrice de confusion normalisée → voir où le modèle se trompe le plus

---

## 🧩 Technologies & Bibliothèques Utilisées

```python
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Utilitaire maison (à adapter si nécessaire)
import fidle.pwk as pwk  # → plot_images, plot_history, plot_confusion_matrix...
