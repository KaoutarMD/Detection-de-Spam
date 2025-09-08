from cassandra.cluster import Cluster

# Connexion à Cassandra
cluster = Cluster(['127.0.0.1'])  # Adresse IP du conteneur
session = cluster.connect()

# Utiliser ou créer un Keyspace (base de données)
session.execute("""
CREATE KEYSPACE IF NOT EXISTS gestionspam
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
""")
session.set_keyspace('gestionspam')

print("Connexion à Cassandra réussie !")
