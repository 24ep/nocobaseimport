import logging
from flask_nocobase_importer.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Helper to convert tuple to dict
def _row_to_dict(row, columns):
    if row:
        return dict(zip(columns, row))
    return None

def add_profile(profile_name: str, db_host: str, db_port: int, db_name: str, db_user: str, db_password: str) -> bool:
    """Adds a new NocoBase connection profile to the database."""
    sql = """
    INSERT INTO nocobase_profiles (profile_name, db_host, db_port, db_name, db_user, db_password)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    db = DatabaseManager()
    try:
        db.execute(sql, (profile_name, db_host, db_port, db_name, db_user, db_password), commit=True)
        logger.info(f"Profile '{profile_name}' added successfully.")
        return True
    except Exception as e:
        logger.error(f"Error adding profile '{profile_name}': {e}")
        return False

def get_all_profiles() -> list:
    """Retrieves all NocoBase connection profiles from the database."""
    sql = "SELECT id, profile_name, db_host, db_port, db_name, db_user, created_at, updated_at FROM nocobase_profiles ORDER BY profile_name ASC"
    # Note: db_password is intentionally not fetched here for list views
    columns = ['id', 'profile_name', 'db_host', 'db_port', 'db_name', 'db_user', 'created_at', 'updated_at']
    db = DatabaseManager()
    try:
        rows = db.fetch_all(sql)
        return [dict(zip(columns, row)) for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Error fetching all profiles: {e}")
        return []

def get_profile_by_id(profile_id: int) -> dict:
    """Retrieves a specific NocoBase connection profile by its ID, including the password."""
    sql = "SELECT id, profile_name, db_host, db_port, db_name, db_user, db_password, created_at, updated_at FROM nocobase_profiles WHERE id = %s"
    columns = ['id', 'profile_name', 'db_host', 'db_port', 'db_name', 'db_user', 'db_password', 'created_at', 'updated_at']
    db = DatabaseManager()
    try:
        row = db.fetch_all(sql, (profile_id,)) # fetch_all returns a list of tuples
        return _row_to_dict(row[0], columns) if row else None
    except Exception as e:
        logger.error(f"Error fetching profile by ID {profile_id}: {e}")
        return None

def update_profile(profile_id: int, profile_name: str, db_host: str, db_port: int, db_name: str, db_user: str, db_password: str) -> bool:
    """Updates an existing NocoBase connection profile."""
    sql = """
    UPDATE nocobase_profiles
    SET profile_name = %s, db_host = %s, db_port = %s, db_name = %s, db_user = %s, db_password = %s
    WHERE id = %s
    """
    # updated_at is handled by the database trigger
    db = DatabaseManager()
    try:
        db.execute(sql, (profile_name, db_host, db_port, db_name, db_user, db_password, profile_id), commit=True)
        logger.info(f"Profile ID {profile_id} ('{profile_name}') updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error updating profile ID {profile_id} ('{profile_name}'): {e}")
        return False

def delete_profile(profile_id: int) -> bool:
    """Deletes a NocoBase connection profile from the database."""
    sql = "DELETE FROM nocobase_profiles WHERE id = %s"
    db = DatabaseManager()
    try:
        db.execute(sql, (profile_id,), commit=True)
        logger.info(f"Profile ID {profile_id} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Error deleting profile ID {profile_id}: {e}")
        return False
