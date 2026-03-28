import json
import os

def save_dashboard_state(filepath, data):
    """Save dashboard configuration to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False

def load_dashboard_state(filepath):
    """Load dashboard configuration from a JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
        return None
