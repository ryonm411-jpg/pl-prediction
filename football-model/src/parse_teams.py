import json
import re
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent
TEAMS_DIR = BASE_DIR / "teams"
OUTPUT_FILE = BASE_DIR / "data" / "teams.json"

def main():
    print(f"Scanning for team profiles in {TEAMS_DIR}...")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    team_files = list(TEAMS_DIR.glob("*.md"))
    all_teams_data = []

    if not team_files:
        print("No team files found.")
        return

    print(f"Found {len(team_files)} team files.")

    for team_file in team_files:
        try:
            print(f"Parsing {team_file.name}...")
            team_data = parse_team_file(team_file)
            if team_data:
                all_teams_data.append(team_data)
        except Exception as e:
            print(f"  [ERROR] Failed to parse {team_file.name}: {e}")

    # Save aggregated data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"teams": all_teams_data}, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_teams_data)} teams to {OUTPUT_FILE}")

def parse_team_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    data = {}

    # --- Header Info ---
    data['team_name'] = extract_field(text, "Team")
    data['league'] = extract_field(text, "League")
    data['season'] = extract_field(text, "Season start")
    
    # If basic info is missing, it might not be a valid team file
    if not data['team_name']:
        print(f"  [WARNING] Could not find 'Team:' field in {file_path.name}. Skipping.")
        return None

    # --- Preferred Formations ---
    # Pattern: - 4-3-3: 0.6
    formations_block = extract_section(text, "## Preferred formations")
    data['preferred_formations'] = parse_key_value_list(formations_block)

    # --- Style Tags ---
    style_block = extract_section(text, "## Style tags")
    data['style_tags'] = parse_simple_list(style_block)

    # --- Strengths ---
    strengths_block = extract_section(text, "## Strengths")
    data['strengths'] = parse_simple_list(strengths_block)

    # --- Weaknesses ---
    weaknesses_block = extract_section(text, "## Weaknesses")
    data['weaknesses'] = parse_simple_list(weaknesses_block)

    # --- Manager Notes ---
    manager_block = extract_section(text, "## Manager notes")
    if manager_block:
        data['manager'] = {
            "name": extract_field(manager_block, "Current manager"),
            "tendencies": extract_field(manager_block, "Tactical tendencies")
        }
    else:
        data['manager'] = {}

    # --- Player Roster ---
    players_block = extract_section(text, "## Player Roster")
    data['roster'] = {
        "key_players": [],
        "rotation": []
    }
    
    if players_block:
        key_players_text = extract_section(players_block, "### Key Players")
        data['roster']['key_players'] = parse_player_list(key_players_text)
        
        rotation_text = extract_section(players_block, "### Squad Rotation")
        data['roster']['rotation'] = parse_player_list(rotation_text)

    return data

# --- Helpers (Reused logic from parse_matches.py) ---

def parse_player_list(text):
    """
    Parses a list of players formatted as:
    - Name: ...
      Position: ...
      ...
    """
    if not text:
        return []
    
    players = []
    # Split by lines starting with "- Name:"
    # We use a lookahead to split but keep the delimiter, or just brute force iterate
    
    # Simple state machine approach
    current_player = {}
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("- Name:"):
            # Save previous player if exists
            if current_player:
                players.append(current_player)
            current_player = {}
            current_player['name'] = line.replace("- Name:", "").strip()
        elif current_player.get('name'): # Only parse other fields if we are inside a player block
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.lower().strip()
                val = val.strip()
                
                if key == "rating":
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                
                current_player[key] = val
    
    # Append the last one
    if current_player:
        players.append(current_player)
        
    return players

def extract_section(text, header_pattern, next_header_pattern=r"\n## "):
    """
    Extracts text between a header and the next header.
    """
    pattern = re.compile(f"{header_pattern}", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    
    start_idx = match.end()
    remaining_text = text[start_idx:]
    
    end_pattern = re.compile(next_header_pattern, re.MULTILINE)
    end_match = end_pattern.search(remaining_text)
    
    if end_match:
        return remaining_text[:end_match.start()].strip()
    else:
        return remaining_text.strip()

def extract_field(text, label_pattern):
    """
    Finds 'Label: Value' in the text.
    """
    pattern = re.compile(f"{label_pattern}:\\s*(.*)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def parse_simple_list(text):
    """
    Extracts lines starting with '- ' as a list of strings.
    """
    if not text:
        return []
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            # removing '- ' and any trailing colon if people put "tag1:"
            cleaned = line[2:].strip().rstrip(':')
            if cleaned:
                items.append(cleaned)
    return items

def parse_key_value_list(text):
    """
    Extracts lines like '- Key: Value' into a dictionary.
    Used for formations (e.g., "- 4-3-3: 0.6")
    """
    if not text:
        return {}
    
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        # Look for "- Key: Value"
        match = re.match(r'-\s*([^:]+):\s*(.*)', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            # Try to convert value to float
            try:
                value = float(value)
            except ValueError:
                pass 
                
            result[key] = value
            
    return result

if __name__ == "__main__":
    main()
