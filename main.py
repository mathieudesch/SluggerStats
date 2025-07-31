import csv
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import requests
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv

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

def create_features(games, rolling_window=10):
    team_stats = defaultdict(lambda: defaultdict(list))
    pitcher_stats = defaultdict(lambda: defaultdict(list))
    featured_games = []
    
    # First pass to build up historical data
    for game in games:
        home_team = game['home_team']
        visitor_team = game['visitor_team']
        home_pitcher = game['home_starting_pitcher_name']
        visitor_pitcher = game['visitor_starting_pitcher_name']

        # Calculate OPS for both teams
        for team in ['home', 'visitor']:
            hits = int(game[f'{team}_hits'])
            walks = int(game[f'{team}_walks'])
            hbp = int(game[f'{team}_hbp'])
            at_bats = int(game[f'{team}_at_bats'])
            sac_flies = int(game[f'{team}_sacrifice_flies'])
            
            doubles = int(game[f'{team}_doubles'])
            triples = int(game[f'{team}_triples'])
            homeruns = int(game[f'{team}_homeruns'])
            singles = hits - (doubles + triples + homeruns)
            
            # Calculate OBP (On-Base Percentage)
            obp_numerator = hits + walks + hbp
            obp_denominator = at_bats + walks + hbp + sac_flies
            obp = obp_numerator / obp_denominator if obp_denominator > 0 else 0
            
            # Calculate SLG (Slugging Percentage)
            slg_numerator = singles + (2 * doubles) + (3 * triples) + (4 * homeruns)
            slg_denominator = at_bats
            slg = slg_numerator / slg_denominator if slg_denominator > 0 else 0
            
            ops = obp + slg
            team_stats[game[f'{team}_team']]['ops'].append(ops)

        # Update team errors
        team_stats[home_team]['errors'].append(int(game['home_errors']))
        team_stats[visitor_team]['errors'].append(int(game['visitor_errors']))
        
        # Update pitcher stats with Team Earned Runs
        pitcher_stats[home_pitcher]['era'].append(int(game['visitor_team_earned_runs']))
        pitcher_stats[visitor_pitcher]['era'].append(int(game['home_team_earned_runs']))
    
    # Second pass to create features
    for game in games:
        home_team = game['home_team']
        visitor_team = game['visitor_team']
        home_pitcher = game['home_starting_pitcher_name']
        visitor_pitcher = game['visitor_starting_pitcher_name']
        
        # Calculate rolling medians for home team
        home_median_ops = np.median(team_stats[home_team]['ops'][-rolling_window:])
        home_median_errors = np.median(team_stats[home_team]['errors'][-rolling_window:])

        # Calculate rolling medians for visitor team
        visitor_median_ops = np.median(team_stats[visitor_team]['ops'][-rolling_window:])
        visitor_median_errors = np.median(team_stats[visitor_team]['errors'][-rolling_window:])
        
        # Calculate rolling median ERA for pitchers
        home_pitcher_era = np.median(pitcher_stats[home_pitcher]['era'][-rolling_window:]) if len(pitcher_stats[home_pitcher]['era']) >= rolling_window else np.nan
        visitor_pitcher_era = np.median(pitcher_stats[visitor_pitcher]['era'][-rolling_window:]) if len(pitcher_stats[visitor_pitcher]['era']) >= rolling_window else np.nan

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
        }
        featured_games.append(featured_game)

    return featured_games


def train_logistic_regression_model(featured_games):
    X = []
    y = []

    for game in featured_games:
        # Check for NaN values and skip the game if any are present
        if any(np.isnan(val) for val in [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era']
        ]):
            continue

        features = [
            game['home_median_ops'],
            game['home_median_errors'],
            game['visitor_median_ops'],
            game['visitor_median_errors'],
            game['home_pitcher_era'],
            game['visitor_pitcher_era'],
        ]
        X.append(features)
        y.append(1 if game['home_score'] > game['visitor_score'] else 0)

    if not X or not y:
        print("Not enough data to train the Logistic Regression model.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model accuracy: {accuracy}")

    return model

def train_xgboost_model(featured_games):
    X = []
    y = []

    for game in featured_games:
        # Check for NaN values and skip the game if any are present
        if any(np.isnan(val) for val in [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era']
        ]):
            continue

        features = [
            game['home_median_ops'],
            game['home_median_errors'],
            game['visitor_median_ops'],
            game['visitor_median_errors'],
            game['home_pitcher_era'],
            game['visitor_pitcher_era'],
        ]
        X.append(features)
        y.append(1 if game['home_score'] > game['visitor_score'] else 0)

    if not X or not y:
        print("Not enough data to train the XGBoost model.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for GridSearchCV - this is the one that yielded 55.28%
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.1, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }

    # Instantiate the XGBoost classifier
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring='accuracy', n_jobs=-1, cv=3, verbose=1)

    # Fit GridSearchCV
    print("Starting hyperparameter tuning with the championship configuration...")
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_xgb_model = grid_search.best_estimator_
    
    print(f"Best XGBoost parameters found: {grid_search.best_params_}")

    # Evaluate the best model on the test set
    y_pred = best_xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fine-tuned XGBoost Model accuracy: {accuracy}")

    return best_xgb_model


def test_model_on_2023_season(featured_games):
    print("\n--- Testing Model Performance on 2023 Season ---")

    # Split data into training (pre-2023) and testing (2023)
    train_games = [game for game in featured_games if not game['date'].startswith('2023')]
    test_games_2023 = [game for game in featured_games if game['date'].startswith('2023')]

    if not test_games_2023:
        print("No 2023 data found to test on.")
        return

    # Prepare training data
    X_train = []
    y_train = []
    for game in train_games:
        if any(np.isnan(val) for val in [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era']
        ]):
            continue
        features = [
            game['home_median_ops'],
            game['home_median_errors'],
            game['visitor_median_ops'],
            game['visitor_median_errors'],
            game['home_pitcher_era'],
            game['visitor_pitcher_era'],
        ]
        X_train.append(features)
        y_train.append(1 if game['home_score'] > game['visitor_score'] else 0)

    # Train the XGBoost model with the championship config
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.1, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring='accuracy', n_jobs=-1, cv=3, verbose=0) # verbose=0 to keep log clean
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Model trained on pre-2023 data.")

    # Prepare test data and make predictions for the 2023 season
    correct_predictions = 0
    total_predictions = 0
    confidence_buckets = {
        "50-55%": {"correct": 0, "total": 0},
        "55-60%": {"correct": 0, "total": 0},
        "60-65%": {"correct": 0, "total": 0},
        "65%+":   {"correct": 0, "total": 0},
    }

    for game in test_games_2023:
        if any(np.isnan(val) for val in [
            game['home_median_ops'], game['home_median_errors'],
            game['visitor_median_ops'], game['visitor_median_errors'],
            game['home_pitcher_era'], game['visitor_pitcher_era']
        ]):
            continue
        
        features = [
            game['home_median_ops'],
            game['home_median_errors'],
            game['visitor_median_ops'],
            game['visitor_median_errors'],
            game['home_pitcher_era'],
            game['visitor_pitcher_era'],
        ]
        actual_winner = 1 if game['home_score'] > game['visitor_score'] else 0
        
        probs = best_model.predict_proba(np.array(features).reshape(1, -1))[0]
        predicted_winner = 1 if probs[1] > probs[0] else 0
        confidence = max(probs)

        if predicted_winner == actual_winner:
            correct_predictions += 1
        total_predictions += 1

        # Sort into confidence buckets
        if 0.50 <= confidence < 0.55:
            bucket = "50-55%"
        elif 0.55 <= confidence < 0.60:
            bucket = "55-60%"
        elif 0.60 <= confidence < 0.65:
            bucket = "60-65%"
        else:
            bucket = "65%+"
        
        confidence_buckets[bucket]['total'] += 1
        if predicted_winner == actual_winner:
            confidence_buckets[bucket]['correct'] += 1

    # Print the final report
    if total_predictions > 0:
        print(f"\nOverall Accuracy on 2023 Season: {correct_predictions / total_predictions:.2%} on {total_predictions} games.")
        print("\nAccuracy by Prediction Confidence:")
        for bucket, data in confidence_buckets.items():
            if data['total'] > 0:
                acc = data['correct'] / data['total']
                print(f"  {bucket}: {acc:.2%} accuracy on {data['total']} games.")
            else:
                print(f"  {bucket}: No games in this confidence range.")
    else:
        print("Could not make any predictions for the 2023 season.")
    print("-------------------------------------------------")


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

def find_value_bets(model, live_odds, featured_games, output_filename):
    if not featured_games:
        print("No featured games to make predictions on.")
        return
    if not live_odds:
        print("No live odds data to process.")
        return

    value_bets = []
    todays_games_summary = []
    tomorrows_games_summary = []
    now = datetime.now(timezone.utc)
    est = ZoneInfo("America/New_York")
    now_est = now.astimezone(est)
    today_date = now_est.date()
    tomorrow_date = today_date + timedelta(days=1)
    
    # Create a lookup map for the last game played between two teams for efficiency
    featured_games_map = {}
    for game in featured_games:
        featured_games_map[(game['home_team'], game['visitor_team'])] = game

    for game_odds in live_odds:
        commence_time = datetime.fromisoformat(game_odds['commence_time'].replace('Z', '+00:00'))
        commence_time_est = commence_time.astimezone(est)
        game_date_est = commence_time_est.date()

        if game_date_est < today_date:
            continue

        home_team_full = game_odds['home_team']
        away_team_full = game_odds['away_team']
        
        home_team_abbr = TEAM_NAME_MAP.get(home_team_full)
        away_team_abbr = TEAM_NAME_MAP.get(away_team_full)

        home_team_odds = "N/A"
        away_team_odds = "N/A"

        # Try to make a prediction and find value bets
        if home_team_abbr and away_team_abbr:
            key = (home_team_abbr, away_team_abbr)
            if key in featured_games_map:
                latest_features = featured_games_map[key]
                
                latest_game_features = np.array([
                    latest_features['home_median_ops'],
                    latest_features['home_median_errors'],
                    latest_features['visitor_median_ops'],
                    latest_features['visitor_median_errors'],
                    latest_features['home_pitcher_era'],
                    latest_features['visitor_pitcher_era'],
                ]).reshape(1, -1)
                
                if not np.isnan(latest_game_features).any():
                    # Get win probabilities for both teams
                    model_probs = model.predict_proba(latest_game_features)[0]
                    home_win_prob = model_probs[1]
                    away_win_prob = model_probs[0]

                    home_team_odds = prob_to_american(home_win_prob)
                    away_team_odds = prob_to_american(away_win_prob)

                    # Determine our predicted winner
                    if home_win_prob > away_win_prob:
                        predicted_winner = home_team_full
                        predicted_winner_prob = home_win_prob
                    else:
                        predicted_winner = away_team_full
                        predicted_winner_prob = away_win_prob

                    # Check for value bets on either team
                    for bookmaker in game_odds['bookmakers']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    bet_on_team = outcome['name']
                                    sportsbook_odds = outcome['price']
                                    sportsbook_prob = decimal_to_prob(sportsbook_odds)
                                    american_odds = decimal_to_american(sportsbook_odds)
                                    
                                    our_prob_for_team = home_win_prob if bet_on_team == home_team_full else away_win_prob
                                    our_model_american_odds = prob_to_american(our_prob_for_team)
                                    
                                    edge = our_prob_for_team - sportsbook_prob

                                    if edge > 0:
                                        value_bets.append({
                                            "home_team": home_team_full,
                                            "away_team": away_team_full,
                                            "predicted_winner": predicted_winner,
                                            "predicted_winner_prob": predicted_winner_prob,
                                            "bet_on_team": bet_on_team,
                                            "our_prob_for_bet": our_prob_for_team,
                                            "sportsbook_prob": sportsbook_prob,
                                            "edge": edge,
                                            "bookmaker": bookmaker['title'],
                                            "price": sportsbook_odds,
                                            "american_odds": american_odds,
                                            "our_model_american_odds": our_model_american_odds,
                                            "game_date": commence_time_est.strftime('%Y-%m-%d %H:%M')
                                        })
                else:
                    print(f"Skipping prediction for {away_team_full} @ {home_team_full}: NaN value in features.")
        
        # Add to daily summary with our model's odds (or N/A if prediction failed)
        game_summary = {
            "home_team": home_team_full,
            "away_team": away_team_full,
            "game_time": commence_time_est.strftime('%H:%M'),
            "home_team_odds": home_team_odds,
            "away_team_odds": away_team_odds
        }

        if game_date_est == today_date:
            todays_games_summary.append(game_summary)
        elif game_date_est == tomorrow_date:
            tomorrows_games_summary.append(game_summary)
    
    # Sort bets by edge
    value_bets.sort(key=lambda x: x['edge'], reverse=True)

    # Save to file
    with open(output_filename, 'w') as f:
        report_time_est = datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Report generated on: {report_time_est} EST\n\n")

        # Today's Summary Section
        f.write("Today's Games and Our Model's Odds\n\n")
        if todays_games_summary:
            todays_games_summary.sort(key=lambda x: x['game_time'])
            for game in todays_games_summary:
                f.write(f"{game['game_time']} EST - {game['home_team']} ({game['home_team_odds']}) vs. {game['away_team']} ({game['away_team_odds']})\n")
        else:
            f.write("No games scheduled for today.\n")
        f.write("\n----------------------------------------\n\n")

        # Tomorrow's Summary Section
        f.write("Tomorrow's Games and Our Model's Odds\n\n")
        if tomorrows_games_summary:
            tomorrows_games_summary.sort(key=lambda x: x['game_time'])
            for game in tomorrows_games_summary:
                f.write(f"{game['game_time']} EST - {game['home_team']} ({game['home_team_odds']}) vs. {game['away_team']} ({game['away_team_odds']})\n")
        else:
            f.write("No games scheduled for tomorrow.\n")
        f.write("\n----------------------------------------\n\n")

        f.write("Model Predictions and Statistical Edges\n\n")
        for bet in value_bets:
            f.write(f"Match: {bet['home_team']} vs {bet['away_team']}\n")
            f.write(f"  Date: {bet['game_date']} EST\n")
            f.write(f"  Our Predicted Winner: {bet['predicted_winner']} ({bet['predicted_winner_prob']:.2%})\n")
            f.write(f"  ----------------------------------------\n")
            f.write(f"  VALUE BET on: {bet['bet_on_team']}\n")
            f.write(f"  Bookmaker: {bet['bookmaker']}\n")
            f.write(f"  Bookmaker Odds: {bet['price']} ({bet['american_odds']})\n")
            f.write(f"  Our Odds: ({bet['our_model_american_odds']})\n")
            f.write(f"  Our Probability: {bet['our_prob_for_bet']:.2%}\n")
            f.write(f"  Bookmaker's Implied Probability: {bet['sportsbook_prob']:.2%}\n")
            f.write(f"  Edge: +{bet['edge']:.2%}\n\n")
    
    print(f"Analysis complete. Results saved to {output_filename}")


if __name__ == "__main__":
    load_dotenv()
    print("Welcome to SluggerStats!")
    
    games = []
    for year in range(2004, 2024):
        file_path = f'data/retrosheet/seasons/{year}/GL{year}.TXT'
        try:
            games.extend(read_gamelog_to_dict(file_path))
        except FileNotFoundError:
            print(f"Warning: Gamelog for {year} not found at {file_path}")
            
    print(f"Loaded {len(games)} games from 2004-2023.")
    
    featured_games = create_features(games)
    
    # Run the new test on the 2023 season
    test_model_on_2023_season(featured_games)

    lr_model = train_logistic_regression_model(featured_games)
    xgb_model = train_xgboost_model(featured_games)

    if lr_model and xgb_model:
        api_key = os.getenv("THE_ODDS_API_KEY")
        if not api_key:
            print("Error: API key not found. Please create a .env file and add your key.")
        else:
            # Check if we have recent odds data
            try:
                with open('data/odds.json', 'r') as f:
                    odds_data = json.load(f)
                # If odds are older than 1 day, fetch new ones
                file_time = datetime.fromtimestamp(os.path.getmtime('data/odds.json'), tz=timezone.utc)
                if (datetime.now(timezone.utc) - file_time).days > 0:
                    print("Odds data is old, fetching new data.")
                    odds_data = get_live_odds(api_key)
            except (FileNotFoundError, json.JSONDecodeError):
                odds_data = get_live_odds(api_key)

            if odds_data:
                print("\n--- Generating predictions for Logistic Regression Model ---")
                find_value_bets(lr_model, odds_data, featured_games, 'model_predictions.txt')
                
                print("\n--- Generating predictions for XGBoost Model ---")
                find_value_bets(xgb_model, odds_data, featured_games, 'xgboost_results.txt') 