import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

print("Loading dataset...")
match_df = pd.read_csv('matches_2008-2024.csv')
delivery_df = pd.read_csv('deliveries_2008-2024.csv')

match_df.columns = match_df.columns.str.strip()
delivery_df.columns = delivery_df.columns.str.strip()

print("Preprocessing dataset...")
team_name_map = {
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Delhi Daredevils': 'Delhi Capitals',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Punjab Kings': 'Kings XI Punjab',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}

# Replace old team names with active ones
for col in ['team1', 'team2', 'toss_winner', 'winner']:
    match_df[col] = match_df[col].replace(team_name_map)

for col in ['batting_team', 'bowling_team']:
    delivery_df[col] = delivery_df[col].replace(team_name_map)

# Keep only active teams
teams = [
    'Chennai Super Kings',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Delhi Capitals',
    'Rajasthan Royals',
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bengaluru',
    'Lucknow Super Giants',
    'Gujarat Titans'
]

match_df = match_df[(match_df['team1'].isin(teams)) & (match_df['team2'].isin(teams))]

# Deal with rain/DLS method matches
if 'method' in match_df.columns:
    match_df = match_df[match_df['method'] != 'D/L']

# Clean up venue/city
match_df['city'] = match_df['city'].fillna('Unknown')
for col in match_df.select_dtypes(include=['object']).columns:
    match_df[col] = match_df[col].astype(str).str.strip()

for col in delivery_df.select_dtypes(include=['object']).columns:
    delivery_df[col] = delivery_df[col].astype(str).str.strip()

# We need columns: match_id, city, winner, target_runs
match_df = match_df[['id', 'city', 'winner', 'target_runs']]

# Merge with deliveries
df = match_df.merge(delivery_df, left_on='id', right_on='match_id')

# We only care about the 2nd innings (chasing)
df = df[df['inning'] == 2].copy()

# Ensure total_runs is numeric
df['total_runs'] = pd.to_numeric(df['total_runs'], errors='coerce').fillna(0)

# Current Score
df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()

# Runs left
df['runs_left'] = df['target_runs'] - df['current_score']

# Balls left
# over is 0-indexed, ball is 1-indexed.
df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
df['balls_left'] = df['balls_left'].apply(lambda x: 0 if x < 0 else x)

# Wickets left
df['is_wicket'] = pd.to_numeric(df['is_wicket'], errors='coerce').fillna(0)
df['wickets_fallen'] = df.groupby('match_id')['is_wicket'].cumsum()
df['wickets_left'] = 10 - df['wickets_fallen']
df['wickets_left'] = df['wickets_left'].apply(lambda x: 0 if x < 0 else x)

# Current Run Rate (CRR)
df['crr'] = (df['current_score'] * 6) / (120 - df['balls_left'])

# Required Run Rate (RRR)
df['rrr'] = (df['runs_left'] * 6) / df['balls_left']
df['rrr'] = df['rrr'].replace([np.inf, -np.inf], 0).fillna(0)

# Target variable (result: 1 if batting_team == winner, else 0)
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

df['result'] = df.apply(result, axis=1)

# Final Dataset
final_df = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'target_runs', 'crr', 'rrr', 'result']]
final_df = final_df.dropna()

print("Dataset prepared for training:", final_df.shape)
X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature engineering / one-hot encoding for categorical vars
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline([
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

print("Training model...")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

output_path = 'predict_win_probability_lr.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(pipe, f)
print(f"Model saved to {output_path}")
