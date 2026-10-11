import json
import re
from pathlib import Path
import sys

# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent
MATCH_ANALYSIS_DIR = BASE_DIR / "match_analysis"
OUTPUT_DIR = BASE_DIR / "data" / "raw_matches"

def main():
    """-
    Main entry point for parsing match files.
    """
    print(f"Scanning for match files in {MATCH_ANALYSIS_DIR}...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Recursively find all .md files
    match_files = list(MATCH_ANALYSIS_DIR.rglob("*.md"))
    
    if not match_files:
        print("No match files found.")
        return

    print(f"Found {len(match_files)} match files.")
    
    for match_file in match_files:
        try:
            print(f"Parsing {match_file.name}...")
            match_data = parse_match_file(match_file)
            
            # Save to JSON
            match_id = match_data['meta'].get('match_id', match_file.stem)
            output_path = OUTPUT_DIR / f"{match_id}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(match_data, f, indent=2, ensure_ascii=False)
                
            print(f"  -> Saved to {output_path.name}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to parse {match_file.name}: {e}")
            import traceback
            traceback.print_exc()

def parse_match_file(file_path):
    """
    Reads a Markdown file and extracts structured data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    data = {
        "meta": {},
        "tactical_phases": {},
        "formations": {},
        "in_match_changes": [],
        "game_plans": {},
        "ratings": {},
        "matchups": [],
        "result_context": {},
        "final_assessment": {},
        "free_notes": ""
    }

    # --- 1. Meta Data ---
    # Extract fields from the top of the file
    data['meta']['match_id'] = extract_field(text, "Match ID")
    data['meta']['date'] = extract_field(text, "Date")
    data['meta']['competition'] = extract_field(text, "Competition")
    data['meta']['venue'] = extract_field(text, "Venue")
    data['meta']['home_team'] = extract_field(text, "Home Team")
    data['meta']['away_team'] = extract_field(text, "Away Team")
    data['meta']['final_score'] = extract_field(text, "Final Score")
    data['meta']['xg'] = extract_field(text, "xG \(est.\)")

    # --- 2. Tactical Phases ---
    phase_1_block = extract_section(text, "### Phase 1 \(0–45\)")
    if phase_1_block:
        data['tactical_phases']['phase_1'] = {
            "control": extract_field(phase_1_block, "Control"),
            "key_structural_factors": extract_field(phase_1_block, "Key Structural Factors"),
            "key_matchups": extract_field(phase_1_block, "Key Matchups"),
            "outcome": extract_field(phase_1_block, "Outcome")
        }
    
    phase_2_block = extract_section(text, "### Phase 2 \(46–90\)")
    if phase_2_block:
        data['tactical_phases']['phase_2'] = {
            "control": extract_field(phase_2_block, "Control"),
            "key_structural_factors": extract_field(phase_2_block, "Key Structural Factors"),
            "key_matchups": extract_field(phase_2_block, "Key Matchups"),
            "outcome": extract_field(phase_2_block, "Outcome")
        }

    # --- 3. Formations ---
    # STOP at next H2 (## ) or separator (---), allow H3 (###) inside
    formations_block = extract_section(text, "## Formations & Structure", next_header_pattern=r"\n##\s|^##\s|---")
    if formations_block:
        # Starting
        starting_block = extract_section(formations_block, "### Starting Formations")
        data['formations']['starting'] = {
            "home": extract_field(starting_block, "Home"),
            "away": extract_field(starting_block, "Away")
        }
        # On-ball
        on_ball_block = extract_section(formations_block, "### On-ball Formations")
        data['formations']['on_ball'] = {
            "home": extract_field(on_ball_block, "Home"),
            "away": extract_field(on_ball_block, "Away"),
            "notes": extract_field(on_ball_block, "Notes")
        }
        # Off-ball
        off_ball_block = extract_section(formations_block, "### Off-ball Formations")
        data['formations']['off_ball'] = {
            "home": extract_field(off_ball_block, "Home"),
            "away": extract_field(off_ball_block, "Away"),
            "notes": extract_field(off_ball_block, "Notes")
        }

    # --- 4. In-match Changes ---
    changes_block = extract_section(text, "## In-match Tactical Changes", next_header_pattern=r"\n##\s|^##\s|---")
    if changes_block:
        # ... (regex for changes remains the same)
        change_matches = re.finditer(r'- Team:\s*(.*?)\n\s*Minute:\s*(.*?)\n\s*Type:\s*(.*?)\n\s*From:\s*(.*?)\n\s*To:\s*(.*?)\n\s*Trigger / Reason:\s*(.*?)\n\s*Immediate Effect:\s*(.*?)\n\s*Effectiveness .*?:\s*(.*?)(?=\n- Team:|\n\n|---|$)', changes_block, re.DOTALL)
        
        for m in change_matches:
            data['in_match_changes'].append({
                "team": m.group(1).strip(),
                "minute": m.group(2).strip(),
                "type": m.group(3).strip(),
                "from": m.group(4).strip(),
                "to": m.group(5).strip(),
                "trigger": m.group(6).strip(),
                "effect": m.group(7).strip(),
                "effectiveness": m.group(8).strip()
            })


    # --- 5. Game Plans ---
    plans_block = extract_section(text, "## Game Plans", next_header_pattern=r"\n##\s|^##\s|---")
    if plans_block:
        data['game_plans']['home'] = parse_team_game_plan(plans_block, "### Home Game Plan")
        data['game_plans']['away'] = parse_team_game_plan(plans_block, "### Away Game Plan")

    # --- 6. Tactical Ratings ---
    ratings_block = extract_section(text, "## Tactical Ratings \(0–100\)", next_header_pattern=r"\n##\s|^##\s|---")
    if ratings_block:
        data['ratings']['home'] = parse_team_ratings(ratings_block, "### Home Team")
        data['ratings']['away'] = parse_team_ratings(ratings_block, "### Away Team")

    # --- 7. Result Context ---
    context_block = extract_section(text, "## Result Context", next_header_pattern=r"\n##\s|^##\s|---")
    if context_block:
        data['result_context']['explained_by_tactics'] = extract_field(context_block, "Did tactics explain the result\?")
        # Parse list manually for overperformance if needed, or just grab the lines
        # Simplification: grab text block
        pass

    # --- 8. Free Notes ---
    data['free_notes'] = extract_section(text, "## Free Notes", next_header_pattern=r"$") # To end of file

    return data

# --- Helpers ---

def extract_section(text, header_pattern, next_header_pattern=r"\n###? |^###? |---"):
    """
    Extracts text between a header and the next header/separator.
    - header_pattern: Regex for the start header.
    - next_header_pattern: Regex for what stops the capture.
    """
    # Find start
    pattern = re.compile(f"{header_pattern}", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    
    start_idx = match.end()
    
    # Find end: Look for next header or '---'
    # We slice from start_idx to avoid finding the current header
    remaining_text = text[start_idx:]
    
    # Improved end pattern: Triple dashes, or hashtags that denote headers
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
    # Regex to handle "Label: Value" where Value can be until end of line
    pattern = re.compile(f"{label_pattern}:\\s*(.*)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def parse_list_items(text):
    """
    Extracts lines starting with '- ' as a list.
    """
    if not text:
        return []
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            items.append(line[2:].strip())
    return items

def parse_team_game_plan(text, team_header):
    """
    Extracts objectives, intentions, strategy for a team.
    """
    team_block = extract_section(text, team_header)
    if not team_block:
        return {}
        
    plan = {}
    plan['objectives'] = parse_list_items(extract_section(team_block, "Primary Objectives:", next_header_pattern=r"\w+:"))
    plan['attacking'] = parse_list_items(extract_section(team_block, "Attacking Intentions:", next_header_pattern=r"\w+:"))
    plan['defensive'] = parse_list_items(extract_section(team_block, "Defensive Intentions:", next_header_pattern=r"\w+:"))
    plan['transition'] = parse_list_items(extract_section(team_block, "Transition Strategy:", next_header_pattern=r"\w+:|$"))
    return plan

def parse_team_ratings(text, team_header):
    """
    Parses the 5 key ratings.
    """
    team_block = extract_section(text, team_header, next_header_pattern=r"### ")
    if not team_block:
        return {}
    
    ratings = {}
    categories = [
        "Pressing Intensity", "Build-up Quality", "Chance Creation", 
        "Defensive Organization", "Defensive Transition Vulnerability"
    ]
    
    for cat in categories:
        # Regex to find the block for this category
        # Pattern: **Category** ... (content) ... until next ** or end
        cat_pattern = re.escape(f"**{cat}**")
        cat_block = extract_section(team_block, cat_pattern, next_header_pattern=r"\*\*")
        
        if cat_block:
            ratings[cat] = {
                "rating": extract_field(cat_block, "Rating"),
                "impact": extract_field(cat_block, "Impact vs Opponent"),
                "certainty": extract_field(cat_block, "Certainty"),
                "reasoning": parse_list_items(extract_section(cat_block, "Reasoning:"))
            }
            
            # Attempt to convert rating to int
            try:
                if ratings[cat]["rating"]:
                    ratings[cat]["rating"] = int(ratings[cat]["rating"])
            except ValueError:
                pass # Keep as string if failed

    return ratings

if __name__ == "__main__":
    main()
