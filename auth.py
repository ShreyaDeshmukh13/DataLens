import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = 'users.db'


def init_db():
    """Initialize the SQLite database and create users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def register_user(username, password):
    """Register a new user with a hashed password."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        hashed_pw = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        conn.close()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, str(e)


def verify_user(username, password):
    """Verify user credentials during login."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row and check_password_hash(row[0], password):
        return True
    return False
