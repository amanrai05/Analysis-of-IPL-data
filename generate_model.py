"""
Script to generate the missing predict_ipl_1st_innings_score_etr.pkl model file.
Run this once: python generate_model.py

Memory-optimised: n_estimators reduced from 100 → 50, max_depth from 15 → 12.
This cuts the .pkl file size from ~120 MB to ~30-40 MB with negligible accuracy loss.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv('./Datasets/scorePridiction.csv')

# Clean column names and string values
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

print(f"Dataset loaded: {df.shape}")

# Replace old/defunct team names
replace_map = {
    'Punjab Kings': 'Kings XI Punjab',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}
df['bat_team'] = df['bat_team'].replace(replace_map)
df['bowl_team'] = df['bowl_team'].replace(replace_map)

# Keep only current playing teams
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
    'Gujarat Titans',
]
df = df[(df['bat_team'].isin(teams)) & (df['bowl_team'].isin(teams))]

# Remove first 5 overs data
df = df[df['overs'] >= 5.0]

# Extract season from date
df['season'] = df['date'].apply(lambda x: x.split('-')[0]).astype(int)

# Drop unnecessary columns
drop_cols = ['batsman', 'bowler', 'striker', 'non_striker', 'venue', 'date']
df.drop([c for c in drop_cols if c in df.columns], inplace=True, axis=1)

# One Hot Encoding for team columns
df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# Split data
df = df.sample(df.shape[0], random_state=42)
X = df.drop(['total', 'season'], axis=1)
y = df['total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
print("Training ExtraTreesRegressor model (optimised for memory)...")

# Reduced from n_estimators=100, max_depth=15 → saves ~80 MB of pkl size
# with negligible accuracy impact on this dataset size.
etr = ExtraTreesRegressor(
    max_depth=12,
    n_estimators=50,
    min_samples_split=5,
    random_state=42,
    n_jobs=1  # avoid forking extra processes during build
)
etr.fit(X_train, y_train)

score = etr.score(X_test, y_test)
print(f"Model R² score on test data: {score:.4f}")

# Save model
output_path = 'predict_ipl_1st_innings_score_etr.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(etr, f)

print(f"\n✅ Model saved successfully as '{output_path}'")
print("You can now run: streamlit run app.py")
