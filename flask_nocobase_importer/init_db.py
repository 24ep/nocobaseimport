import os
import logging
# Ensure db_manager can be imported. If FLASK_APP context is needed for env vars,
# this script might need to be run via 'flask exec' or similar,
# or ensure env vars are loaded/available when run directly.
from flask_nocobase_importer.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initializes the importer application's database by creating necessary tables
    if they don't already exist.
    """
    logger.info("Attempting to initialize the importer database...")
    
    db = DatabaseManager() # Assumes DB_HOST etc. env vars point to importer_db
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS nocobase_profiles (
        id SERIAL PRIMARY KEY,
        profile_name VARCHAR(255) UNIQUE NOT NULL,
        db_host VARCHAR(255) NOT NULL,
        db_port INTEGER NOT NULL,
        db_name VARCHAR(255) NOT NULL,
        db_user VARCHAR(255) NOT NULL,
        db_password TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_trigger_function_sql = """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
       NEW.updated_at = NOW();
       RETURN NEW;
    END;
    $$ LANGUAGE 'plpgsql';
    """

    create_trigger_sql = """
    DO $$
    BEGIN
       IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_nocobase_profiles_updated_at' AND tgrelid = 'nocobase_profiles'::regclass) THEN
          CREATE TRIGGER trigger_update_nocobase_profiles_updated_at
          BEFORE UPDATE ON nocobase_profiles
          FOR EACH ROW
          EXECUTE FUNCTION update_updated_at_column();
       END IF;
    END $$;
    """

    conn = None  # Initialize conn to None
    try:
        conn = db.get_connection()
        with conn.cursor() as cur:
            logger.info("Creating 'nocobase_profiles' table if it doesn't exist...")
            cur.execute(create_table_sql)
            
            logger.info("Creating or replacing 'update_updated_at_column' trigger function...")
            cur.execute(create_trigger_function_sql)
            
            logger.info("Applying 'updated_at' trigger to 'nocobase_profiles' table if not exists...")
            cur.execute(create_trigger_sql)
            
            conn.commit()
        logger.info("Database initialization successful: 'nocobase_profiles' table and 'updated_at' trigger are set up.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # To run this script directly for local development (ensure .env is loaded for db_manager):
    # Example:
    # from dotenv import load_dotenv
    # import sys
    # # Assuming this script is in flask_nocobase_importer directory, and .env is in project root
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)
    # load_dotenv(os.path.join(project_root, '.env'))

    logger.info("Running database initializer script...")
    initialize_database()
    logger.info("Database initializer script finished.")
