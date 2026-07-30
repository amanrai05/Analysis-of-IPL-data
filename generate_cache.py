import pandas as pd

# Columns actually used across all pages (after stripping)
MATCHES_COLS = [
    'id', 'season', 'city', 'venue', 'team1', 'team2',
    'toss_winner', 'toss_decision', 'winner', 'result',
    'result_margin', 'player_of_match', 'target_runs',
    'umpire1', 'umpire2'
]

DELIVERIES_COLS = [
    'match_id', 'inning', 'batting_team', 'bowling_team',
    'over', 'ball', 'batter', 'bowler', 'non_striker',
    'batsman_runs', 'extra_runs', 'total_runs', 'extras_type', 'is_wicket'
]

# Updating the team names
def latest_teams(df, cols):
    # mapping old to latest
    team_name_map = {
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Delhi Daredevils': 'Delhi Capitals',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Punjab Kings': 'Kings XI Punjab',
        'Rising Pune Supergiants': 'Rising Pune Supergiant'
    }

    # Replace old team names with the latest names
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].replace(team_name_map)

    return df

# Updating the venue names
def unique_stadium(matches_df):
    venue_map = {
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Eden Gardens, Kolkata': 'Eden Gardens',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
        'M Chinnaswamy Stadium, Bengalore': 'M Chinnaswamy Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium'
    }
    matches_df['venue'] = matches_df['venue'].replace(venue_map)

def trimSpaceInValues(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df

# Optimize Memory Usage
def optimize_memory(df):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype('category')
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

print("Generating cached dataframes to speed up Streamlit loading...")

print("Loading matches...")
df_matches = pd.read_csv('matches_2008-2024.csv')
df_matches.columns = df_matches.columns.str.strip()
cols_to_keep_matches = [c for c in MATCHES_COLS if c in df_matches.columns]
df_matches = df_matches[cols_to_keep_matches]
df_matches = trimSpaceInValues(df_matches)
df_matches = latest_teams(df_matches, ['team1', 'team2', 'toss_winner', 'winner'])
unique_stadium(df_matches)
df_matches = optimize_memory(df_matches)
df_matches.to_pickle('matches_cached.pkl')
print("Saved matches_cached.pkl")

print("Loading deliveries...")
df_deliveries = pd.read_csv('deliveries_2008-2024.csv')
df_deliveries.columns = df_deliveries.columns.str.strip()
cols_to_keep_deliveries = [c for c in DELIVERIES_COLS if c in df_deliveries.columns]
df_deliveries = df_deliveries[cols_to_keep_deliveries]
df_deliveries = trimSpaceInValues(df_deliveries)

if 'extras_type' in df_deliveries.columns:
    df_deliveries.loc[df_deliveries['extras_type'].str.strip() == '', 'extras_type'] = 'None'

df_deliveries = latest_teams(df_deliveries, ['batting_team', 'bowling_team'])
df_deliveries = optimize_memory(df_deliveries)
df_deliveries.to_pickle('deliveries_cached.pkl')
print("Saved deliveries_cached.pkl")

print("Data caching complete!")
