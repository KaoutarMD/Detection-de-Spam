from cassandra.cluster import Cluster
import csv
import uuid

# Connexion au cluster Cassandra
cluster = Cluster(['127.0.0.1'])
session = cluster.connect()
session.set_keyspace('gestionspam')

# Création de la table `messages`
session.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY,
        label TEXT,
        content TEXT
    )
""")
print("Table 'messages' créée avec succès.")

# Fonction pour lire le fichier CSV et insérer les données dans Cassandra
def insert_data_to_cassandra(file_path):
    # Utiliser un encodage compatible
    with open(file_path, mode='r', encoding='latin-1') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Ignorer l'en-tête du fichier CSV si présent

        for row in reader:
            # Vérifier si les lignes sont valides
            if len(row) >= 2:
                message_id = uuid.uuid4()  # Générer un UUID unique
                label = row[0].strip()  # Première colonne : "ham" ou "spam"
                content = row[1].strip()  # Deuxième colonne : contenu du message

                # Insérer les données dans la table `messages`
                session.execute("""
                    INSERT INTO messages (id, label, content) VALUES (%s, %s, %s)
                """, (message_id, label, content))
                print(f"Données insérées : Label: {label}, Content: {content}")

# Lire et insérer les données depuis le fichier CSV
insert_data_to_cassandra('spam.csv')

