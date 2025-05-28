import os # Add this import
import psycopg2

# ---------------------------
# Database Manager
# ---------------------------
class DatabaseManager:
    def __init__(self,
                 host=None, # Allow parameters to be passed for testing, but prioritize env vars
                 port=None,
                 database=None,
                 user=None,
                 password=None):
        self.config = {
            "host": os.environ.get("DB_HOST", host or "10.0.10.25"),
            "port": int(os.environ.get("DB_PORT", port or 18088)), # Ensure port is an integer
            "database": os.environ.get("DB_NAME", database or "nocobase"), # Changed from DB_DATABASE to DB_NAME
            "user": os.environ.get("DB_USER", user or "nocobase"),
            "password": os.environ.get("DB_PASSWORD", password or "nocobase")
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
