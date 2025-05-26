import psycopg2

# ---------------------------
# Database Manager
# ---------------------------
class DatabaseManager:
    def __init__(self, host="10.0.10.25", port=18088, database="nocobase", user="nocobase", password="nocobase"):
        self.config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }

    def get_connection(self):
        return psycopg2.connect(**self.config)

    def fetch_all(self, query, params=None):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def execute(self, query, params=None, commit=False, fetchone=False):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if commit:
                    conn.commit()
                if fetchone:
                    return cur.fetchone()
                return None

    def execute_many(self, query, params_list):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                conn.commit()
                return cur.rowcount
