import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import csv
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import requests
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

def get_advanced_stats(api_key, season, stat_type="batting"):
    """Fetches advanced stats for a given season from SportsBlaze."""
    url = f"https://api.sportsblaze.com/mlb/v1/stats/advanced/{season}/{stat_type}.json?key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        stats = response.json()
        player_stats = {player['name']: player['stats'] for player in stats.get('players', [])}
        return player_stats
    except requests.exceptions.RequestException as e:
        print(f"Error fetching advanced stats: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from advanced stats response: {response.text}")
        return {}

def get_daily_schedule(api_key, game_date):
    """Fetches the daily schedule from the SportsBlaze API."""
    url = f"https://api.sportsblaze.com/mlb/v1/schedule/daily/{game_date.strftime('%Y-%m-%d')}.json?key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching daily schedule: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from daily schedule response: {response.text}")
        return None

TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHN",
    "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA",
    "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYN",
    "New York Yankees": "NYA",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDN",
    "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}

GAMELOG_HEADERS = [
    "date", "game_num", "day_of_week", "visitor_team", "visitor_league",
    "visitor_game_num", "home_team", "home_league", "home_game_num",
    "visitor_score", "home_score", "game_length_outs", "day_night",
    "completion_info", "forfeit_info", "protest_info", "park_id",
    "attendance", "game_length_minutes", "visitor_line_score",
    "home_line_score", "visitor_at_bats", "visitor_hits", "visitor_doubles",
    "visitor_triples", "visitor_homeruns", "visitor_rbi",
    "visitor_sacrifice_hits", "visitor_sacrifice_flies", "visitor_hbp",
    "visitor_walks", "visitor_intentional_walks", "visitor_strikeouts",
    "visitor_stolen_bases", "visitor_caught_stealing",
    "visitor_grounded_into_double", "visitor_catchers_interference",
    "visitor_left_on_base", "visitor_pitchers_used",
    "visitor_individual_earned_runs", "visitor_team_earned_runs",
    "visitor_wild_pitches", "visitor_balks", "visitor_putouts",
    "visitor_assists", "visitor_errors", "visitor_passed_balls",
    "visitor_double_plays", "visitor_triple_plays", "home_at_bats",
    "home_hits", "home_doubles", "home_triples", "home_homeruns", "home_rbi",
    "home_sacrifice_hits", "home_sacrifice_flies", "home_hbp", "home_walks",
    "home_intentional_walks", "home_strikeouts", "home_stolen_bases",
    "home_caught_stealing", "home_grounded_into_double",
    "home_catchers_interference", "home_left_on_base",
    "home_pitchers_used", "home_individual_earned_runs",
    "home_team_earned_runs", "home_wild_pitches", "home_balks",

    "home_putouts", "home_assists", "home_errors", "home_passed_balls",
    "home_double_plays", "home_triple_plays", "home_plate_umpire_id",
    "home_plate_umpire_name", "1b_umpire_id", "1b_umpire_name",
    "2b_umpire_id", "2b_umpire_name", "3b_umpire_id", "3b_umpire_name",
    "lf_umpire_id", "lf_umpire_name", "rf_umpire_id", "rf_umpire_name",
    "visitor_manager_id", "visitor_manager_name", "home_manager_id",
    "home_manager_name", "winning_pitcher_id", "winning_pitcher_name",
    "losing_pitcher_id", "losing_pitcher_name", "saving_pitcher_id",
    "saving_pitcher_name", "game_winning_rbi_batter_id",
    "game_winning_rbi_batter_name", "visitor_starting_pitcher_id",
    "visitor_starting_pitcher_name", "home_starting_pitcher_id",
    "home_starting_pitcher_name",
    # Starting players up to 9 for each team
    "visitor_player_1_id", "visitor_player_1_name", "visitor_player_1_pos",
    "visitor_player_2_id", "visitor_player_2_name", "visitor_player_2_pos",
    "visitor_player_3_id", "visitor_player_3_name", "visitor_player_3_pos",
    "visitor_player_4_id", "visitor_player_4_name", "visitor_player_4_pos",
    "visitor_player_5_id", "visitor_player_5_name", "visitor_player_5_pos",
    "visitor_player_6_id", "visitor_player_6_name", "visitor_player_6_pos",
    "visitor_player_7_id", "visitor_player_7_name", "visitor_player_7_pos",
    "visitor_player_8_id", "visitor_player_8_name", "visitor_player_8_pos",
    "visitor_player_9_id", "visitor_player_9_name", "visitor_player_9_pos",
    "home_player_1_id", "home_player_1_name", "home_player_1_pos",
    "home_player_2_id", "home_player_2_name", "home_player_2_pos",
    "home_player_3_id", "home_player_3_name", "home_player_3_pos",
    "home_player_4_id", "home_player_4_name", "home_player_4_pos",
    "home_player_5_id", "home_player_5_name", "home_player_5_pos",
    "home_player_6_id", "home_player_6_name", "home_player_6_pos",
    "home_player_7_id", "home_player_7_name", "home_player_7_pos",
    "home_player_8_id", "home_player_8_name", "home_player_8_pos",
    "home_player_9_id", "home_player_9_name", "home_player_9_pos",
    "additional_info", "acquisition_info"
]


def read_gamelog_to_dict(file_path):
    games = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            game_dict = dict(zip(GAMELOG_HEADERS, row))
            games.append(game_dict)
    return games

def create_features(games, rolling_window=100):
    team_stats = defaultdict(lambda: defaultdict(list))
    pitcher_stats = defaultdict(lambda: defaultdict(list))
    featured_games = []

    # Sort games by date to process them in chronological order
    games.sort(key=lambda x: x['date'])

    for game in games:
        home_team = game['home_team']
        visitor_team = game['visitor_team']
        home_pitcher = game['home_starting_pitcher_name']
        visitor_pitcher = game['visitor_starting_pitcher_name']

        #
        # Create features using data *before* the current game
        #
        home_median_ops = np.median(team_stats[home_team]['ops'][-rolling_window:]) if team_stats[home_team]['ops'] else np.nan
        home_median_errors = np.median(team_stats[home_team]['errors'][-rolling_window:]) if team_stats[home_team]['errors'] else np.nan
        visitor_median_ops = np.median(team_stats[visitor_team]['ops'][-rolling_window:]) if team_stats[visitor_team]['ops'] else np.nan
        visitor_median_errors = np.median(team_stats[visitor_team]['errors'][-rolling_window:]) if team_stats[visitor_team]['errors'] else np.nan
        
        home_pitcher_era = np.median(pitcher_stats[home_pitcher]['era'][-rolling_window:]) if pitcher_stats[home_pitcher]['era'] else np.nan
        visitor_pitcher_era = np.median(pitcher_stats[visitor_pitcher]['era'][-rolling_window:]) if pitcher_stats[visitor_pitcher]['era'] else np.nan
        home_team_era = np.median(team_stats[home_team]['team_era'][-rolling_window:]) if team_stats[home_team]['team_era'] else np.nan
        visitor_team_era = np.median(team_stats[visitor_team]['team_era'][-rolling_window:]) if team_stats[visitor_team]['team_era'] else np.nan

        featured_game = {
            'date': game['date'],
            'home_team': home_team,
            'visitor_team': visitor_team,
            'home_score': int(game['home_score']),
            'visitor_score': int(game['visitor_score']),
            'home_starting_pitcher': home_pitcher,
            'visitor_starting_pitcher': visitor_pitcher,
            'home_median_ops': home_median_ops,
            'home_median_errors': home_median_errors,
            'visitor_median_ops': visitor_median_ops,
            'visitor_median_errors': visitor_median_errors,
            'home_pitcher_era': home_pitcher_era,
            'visitor_pitcher_era': visitor_pitcher_era,
            'home_team_era': home_team_era,
            'visitor_team_era': visitor_team_era,
        }
        featured_games.append(featured_game)
        
        # Now, update the stats with the outcome of the *current* game for future calculations
        #
        for team_type in ['home', 'visitor']:
            team_abbr = game[f'{team_type}_team']
            
            # Calculate OPS
            hits = int(game[f'{team_type}_hits'])
            walks = int(game[f'{team_type}_walks'])
            hbp = int(game[f'{team_type}_hbp'])
            at_bats = int(game[f'{team_type}_at_bats'])
            sac_flies = int(game[f'{team_type}_sacrifice_flies'])
            doubles = int(game[f'{team_type}_doubles'])
            triples = int(game[f'{team_type}_triples'])
            homeruns = int(game[f'{team_type}_homeruns'])
            singles = hits - (doubles + triples + homeruns)
            
            obp_numerator = hits + walks + hbp
            obp_denominator = at_bats + walks + hbp + sac_flies
            obp = obp_numerator / obp_denominator if obp_denominator > 0 else 0
            
            slg_numerator = singles + (2 * doubles) + (3 * triples) + (4 * homeruns)
            slg_denominator = at_bats
            slg = slg_numerator / slg_denominator if slg_denominator > 0 else 0
            
            ops = obp + slg
            team_stats[team_abbr]['ops'].append(ops)

            # Store errors
            team_stats[team_abbr]['errors'].append(int(game[f'{team_type}_errors']))
            
            # Store team earned runs for team-based ERA proxy
            team_stats[team_abbr]['team_era'].append(int(game[f'{team_type}_team_earned_runs']))
            
            # Store runs allowed for team-based ERA proxy
            opponent_type = 'visitor' if team_type == 'home' else 'home'
            runs_allowed = int(game[f'{opponent_type}_score'])
            team_stats[team_abbr]['era_proxy'].append(runs_allowed)

        # Update pitcher-specific stats
        pitcher_stats[home_pitcher]['era'].append(int(game['visitor_team_earned_runs']))
        pitcher_stats[visitor_pitcher]['era'].append(int(game['home_team_earned_runs']))

    return featured_games, team_stats, pitcher_stats






def train_and_evaluate_model(featured_games, test_year=2024):
    print(f"\n--- Training on pre-{test_year} data, testing on {test_year} season ---")

    # Split data into training and testing based on the test_year
    train_games = [game for game in featured_games if not game['date'].startswith(str(test_year))]
    test_games = [game for game in featured_games if game['date'].startswith(str(test_year))]

    if not test_games:
        print(f"No {test_year} data found to test on.")
        return None

    # Prepare training data
    X_train, y_train = [], []
    for game in train_games:
        if any(np.isnan(val) for val in [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era'],
            game['home_team_era'], game['visitor_team_era']
        ]):
            continue
        features = [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era'],
            game['home_team_era'], game['visitor_team_era'],
        ]
        X_train.append(features)
        y_train.append(1 if game['home_score'] > game['visitor_score'] else 0)

    if not X_train:
        print("Not enough training data available.")
        return None

    # Train the XGBoost model with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.1, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring='accuracy', n_jobs=-1, cv=3, verbose=1)
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best XGBoost parameters found: {grid_search.best_params_}")
    print(f"Model trained on {len(X_train)} pre-{test_year} games.")

    # Prepare test data and make predictions for the test season
    correct_predictions, total_predictions = 0, 0
    confidence_buckets = {
        "50-55%": {"correct": 0, "total": 0}, "55-60%": {"correct": 0, "total": 0},
        "60-65%": {"correct": 0, "total": 0}, "65%+":   {"correct": 0, "total": 0},
    }

    for game in test_games:
        if any(np.isnan(val) for val in [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era'],
            game['home_team_era'], game['visitor_team_era']
        ]):
            continue
        
        features = [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era'],
            game['home_team_era'], game['visitor_team_era'],
        ]
        actual_winner = 1 if game['home_score'] > game['visitor_score'] else 0
        
        probs = best_model.predict_proba(np.array(features).reshape(1, -1))[0]
        predicted_winner = 1 if probs[1] > probs[0] else 0
        confidence = max(probs)

        if predicted_winner == actual_winner:
            correct_predictions += 1
        total_predictions += 1

        if 0.50 <= confidence < 0.55: bucket = "50-55%"
        elif 0.55 <= confidence < 0.60: bucket = "55-60%"
        elif 0.60 <= confidence < 0.65: bucket = "60-65%"
        else: bucket = "65%+"
        
        confidence_buckets[bucket]['total'] += 1
        if predicted_winner == actual_winner:
            confidence_buckets[bucket]['correct'] += 1

    if total_predictions > 0:
        print(f"\nOverall Accuracy on {test_year} Season: {correct_predictions / total_predictions:.2%} on {total_predictions} games.")
        print("\nAccuracy by Prediction Confidence:")
        for bucket, data in confidence_buckets.items():
            if data['total'] > 0:
                acc = data['correct'] / data['total']
                print(f"  {bucket}: {acc:.2%} accuracy on {data['total']} games.")
            else:
                print(f"  {bucket}: No games in this confidence range.")
    else:
        print(f"Could not make any predictions for the {test_year} season.")
    print("-------------------------------------------------")
    
    return best_model


def get_live_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={api_key}&regions=us&markets=h2h"
    response = requests.get(url)
    
    if response.status_code == 200:
        odds_data = response.json()
        with open('data/odds.json', 'w') as f:
            json.dump(odds_data, f, indent=4)
        print("Live odds data saved to data/odds.json")
        return odds_data
    else:
        print(f"Failed to get odds: {response.status_code}")
        return None

def decimal_to_prob(decimal_odds):
    return 1 / decimal_odds

def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"

def prob_to_american(prob):
    if prob <= 0 or prob >= 1:
        return "N/A"
    decimal_odds = 1 / prob
    return decimal_to_american(decimal_odds)

def find_value_bets(model, api_key, team_stats, pitcher_stats, output_filename, rolling_window=100):
    now = datetime.now(timezone.utc)
    est = ZoneInfo("America/New_York")
    now_est = now.astimezone(est)
    today_date = now_est.date()
    current_season = today_date.year

    # Fetch today's and tomorrow's schedules
    todays_schedule = get_daily_schedule(api_key, today_date)
    tomorrows_schedule = get_daily_schedule(api_key, today_date + timedelta(days=1))
    
    live_games = []
    if todays_schedule and 'games' in todays_schedule:
        live_games.extend(todays_schedule['games'])
    if tomorrows_schedule and 'games' in tomorrows_schedule:
        live_games.extend(tomorrows_schedule['games'])

    if not live_games:
        print("No live games found for today or tomorrow.")
        return

    games_summary = []

    for game in live_games:
        commence_time = datetime.fromisoformat(game['date'].replace('Z', '+00:00'))
        commence_time_est = commence_time.astimezone(est)

        home_team_full = game['teams']['home']['name']
        away_team_full = game['teams']['away']['name']
        
        home_team_abbr = TEAM_NAME_MAP.get(home_team_full)
        away_team_abbr = TEAM_NAME_MAP.get(away_team_full)

        home_team_odds, away_team_odds = "N/A", "N/A"

        if home_team_abbr and away_team_abbr:
            home_pitcher = game.get('starters', {}).get('home', {}).get('name') or "Unknown"
            away_pitcher = game.get('starters', {}).get('away', {}).get('name') or "Unknown"

            # The daily schedule endpoint doesn't seem to include lineups based on docs.
            # We will use historical median OPS as the primary offensive feature.
            home_ops = np.median(team_stats[home_team_abbr]['ops'][-rolling_window:])
            visitor_ops = np.median(team_stats[away_team_abbr]['ops'][-rolling_window:])
            
            home_median_errors = np.median(team_stats[home_team_abbr]['errors'][-rolling_window:])
            visitor_median_errors = np.median(team_stats[away_team_abbr]['errors'][-rolling_window:])
            
            home_pitcher_era = np.median(pitcher_stats[home_pitcher]['era'][-rolling_window:]) if home_pitcher != "Unknown" and pitcher_stats[home_pitcher]['era'] else np.median(team_stats[home_team_abbr]['team_era'][-rolling_window:])
            visitor_pitcher_era = np.median(pitcher_stats[away_pitcher]['era'][-rolling_window:]) if away_pitcher != "Unknown" and pitcher_stats[away_pitcher]['era'] else np.median(team_stats[away_team_abbr]['team_era'][-rolling_window:])
            
            home_team_era = np.median(team_stats[home_team_abbr]['team_era'][-rolling_window:])
            visitor_team_era = np.median(team_stats[away_team_abbr]['team_era'][-rolling_window:])
            
            features = np.array([
                home_ops, home_median_errors,
                visitor_ops, visitor_median_errors,
                home_pitcher_era, visitor_pitcher_era,
                home_team_era, visitor_team_era,
            ]).reshape(1, -1)

            if not np.isnan(features).any():
                model_probs = model.predict_proba(features)[0]
                home_win_prob, away_win_prob = model_probs[1], model_probs[0]
                home_team_odds = prob_to_american(home_win_prob)
                away_team_odds = prob_to_american(away_win_prob)
            
            games_summary.append({
                "home_team": home_team_full, "away_team": away_team_full,
                "game_time": commence_time_est.strftime('%H:%M'),
                "home_team_odds": home_team_odds, "away_team_odds": away_team_odds,
                "home_pitcher": home_pitcher, "away_pitcher": away_pitcher,
                "date": commence_time_est.date()
            })

    with open(output_filename, 'w') as f:
        report_time_est = datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Report generated on: {report_time_est} EST\n\n")

        f.write("Today's Games and Our Model's Odds\n\n")
        today_games = [g for g in games_summary if g['date'] == today_date]
        if today_games:
            for game in sorted(today_games, key=lambda x: x['game_time']):
                f.write(f"{game['game_time']} EST - {game['home_team']} ({game['home_team_odds']}) vs. {game['away_team']} ({game['away_team_odds']})\n")
                f.write(f"  Pitchers: {game['home_pitcher']} vs {game['away_pitcher']}\n\n")
        else:
            f.write("No games scheduled for today.\n")
        
        f.write("\n----------------------------------------\n\n")

        f.write("Tomorrow's Games and Our Model's Odds\n\n")
        tomorrow_games = [g for g in games_summary if g['date'] != today_date]
        if tomorrow_games:
            for game in sorted(tomorrow_games, key=lambda x: x['game_time']):
                f.write(f"{game['game_time']} EST - {game['home_team']} ({game['home_team_odds']}) vs. {game['away_team']} ({game['away_team_odds']})\n")
                f.write(f"  Pitchers: {game['home_pitcher']} vs {game['away_pitcher']}\n\n")
        else:
            f.write("No games scheduled for tomorrow.\n")

    print(f"Analysis complete. Results saved to {output_filename}")


if __name__ == "__main__":
    print("Welcome to SluggerStats!")
    
    games = []
    for year in range(2002, 2025):
        file_path = f'data/retrosheet/seasons/{year}/GL{year}.TXT'
        try:
            games.extend(read_gamelog_to_dict(file_path))
        except FileNotFoundError:
            print(f"Warning: Gamelog for {year} not found at {file_path}")
            
    print(f"Loaded {len(games)} games from 2002-2024.")
    
    featured_games, team_stats, pitcher_stats = create_features(games)
    
    # Train the model on historical data and evaluate it on the most recent season
    xgb_model = train_and_evaluate_model(featured_games, test_year=2024)

    if xgb_model:
        api_key = os.getenv("SPORTSBLAZE_API_KEY")
        if not api_key:
            print("Error: SPORTSBLAZE_API_KEY not found in .env file.")
        else:
            print("\n--- Generating predictions for XGBoost Model ---")
            find_value_bets(xgb_model, api_key, team_stats, pitcher_stats, 'xgboost_results.txt')
