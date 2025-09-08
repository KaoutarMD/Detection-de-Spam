# 📧 Spam Detection avec Python et Cassandra

## 📌 Contexte

Le spam (courriel indésirable) est un problème majeur dans les communications numériques.
Ce projet vise à développer un **système de détection de spam** robuste, capable de classer automatiquement les e-mails en **spam** ou **non-spam**, avec un stockage et une analyse efficaces grâce à **Cassandra**.

---

## 🎯 Objectifs

* Détecter et classifier les e-mails en spam ou non-spam.
* Gérer et stocker de grandes quantités de données dans **Cassandra**.
* Fournir une **API en temps réel** pour la prédiction via **Flask**.

---

## 🛠️ Technologies utilisées

* **Python** :

  * NLTK, spaCy → Prétraitement du langage naturel.
  * Scikit-learn → Modèles de base (Naïve Bayes).
  * Transformers (Hugging Face - DistilBERT) → Modèle avancé.
  * Flask → API web pour la prédiction en temps réel.

* **Cassandra** :

  * Stockage distribué et scalable des e-mails et prédictions.
  * Gestion des données historiques pour le ré-entraînement des modèles.

* **Docker** :

  * Déploiement de Cassandra et des services associés.
  * Fichier `docker-compose.yml` pour lancer facilement l’environnement.

---

## 🏗️ Architecture du projet

1. **Collecte & Stockage** → Insertion des e-mails et métadonnées dans Cassandra.
2. **Prétraitement** → Nettoyage, lemmatisation et vectorisation du texte.
3. **Entraînement des modèles** :

   * Naïve Bayes (baseline).
   * DistilBERT (modèle avancé).
4. **API Flask** → Endpoint `/predict` pour classer un e-mail en temps réel.
5. **Stockage des prédictions** → Résultats sauvegardés dans Cassandra.

---

## 🚀 Installation et exécution

### 1️⃣ Prérequis

* Python 3.x
* Docker & Docker Compose
* Cassandra Driver :

  ```bash
  pip install cassandra-driver
  ```

### 2️⃣ Lancer Cassandra avec Docker

```bash
docker pull cassandra:latest
docker run --name cassandra -d -p 9042:9042 cassandra
```

ou avec **Docker Compose** :

```bash
docker-compose up -d
```

### 3️⃣ Importer les données dans Cassandra

Créer la table `messages` :

```sql
CREATE KEYSPACE gestionspam WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
USE gestionspam;
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    label TEXT,
    content TEXT
);
```

Puis importer les données CSV via un script Python.

### 4️⃣ Lancer l’application Flask

```bash
python app.py
```

L’API sera disponible sur : `http://localhost:5000`

---

## 🌐 API Endpoints

* `GET /` → Page d’accueil.
* `POST /predict`

  * Input : JSON contenant le texte de l’e-mail.
  * Output : Prédiction (`Spam` ou `Non Spam`) + probabilité.

Exemple :

```json
{
  "email": "Congratulations! You've won a free iPhone!"
}
```

Réponse :

```json
{
  "prediction": "Spam",
  "probability": 0.95
}
```

---

## ✅ Fonctionnalités principales

* Détection de spam en temps réel.
* Stockage des e-mails et prédictions dans Cassandra.
* Réentraînement automatique avec de nouvelles données.
* Haute disponibilité et scalabilité grâce à Cassandra.

---

## 📈 Bénéfices

* Haute précision grâce à DistilBERT.
* Stockage scalable avec Cassandra.
* Réponse rapide via API Flask.
* Évolutivité et intégration facile avec d’autres services.

---

## 👨‍💻 Auteurs

Projet réalisé dans le cadre du module **NoSQL - FST Fès**.

---

