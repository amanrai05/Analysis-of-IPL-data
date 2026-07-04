import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from scrollToTop import create_scroll_to_top_button

# Load the model
try:
    with open('predict_win_probability_lr.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    pipe = None
    
teams = [
    'Chennai Super Kings', 'Kolkata Knight Riders', 'Kings XI Punjab',
    'Delhi Capitals', 'Rajasthan Royals', 'Sunrisers Hyderabad',
    'Mumbai Indians', 'Royal Challengers Bengaluru', 'Lucknow Super Giants',
    'Gujarat Titans'
]

cities = [
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 'Cape Town',
    'Centurion', 'Chandigarh', 'Chennai', 'Cuttack', 'Delhi', 'Dharamsala', 'Dubai',
    'Durban', 'East London', 'Guwahati', 'Hyderabad', 'Indore', 'Jaipur', 'Johannesburg',
    'Kanpur', 'Kimberley', 'Kochi', 'Kolkata', 'Lucknow', 'Mohali', 'Mumbai', 'Nagpur',
    'Navi Mumbai', 'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 'Sharjah',
    'Unknown', 'Visakhapatnam'
]

def app():
    st.markdown('''
    <h1 style='text-align:center; color: #700961;'><strong> 🎲 PREDICTING WIN PROBABILITY FOR A TEAM 🎲</strong></h1>
    <hr style="border-top: 3px solid #700961;">
    ''', unsafe_allow_html=True)
    
    if pipe is None:
        st.error("Model file 'predict_win_probability_lr.pkl' not found. Please train the model first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the Batting Team (Chasing)', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the Bowling Team (Defending)', sorted(teams))

    city = st.selectbox('Select the Host City', sorted(cities))
    
    col3, col4, col5 = st.columns(3)
    with col3:
        target = st.number_input('Target Score', min_value=0, value=150, step=1)
    with col4:
        score = st.number_input('Current Score', min_value=0, max_value=int(target), value=50, step=1)
    with col5:
        wickets = st.number_input('Wickets Out', min_value=0, max_value=10, value=2, step=1)
        
    overs = st.number_input('Overs Completed (e.g., 5.3 means 5 overs and 3 balls)', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    
    if st.button('Predict Win Probability', use_container_width=True):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams cannot be the same!")
        else:
            runs_left = target - score
            
            # Convert overs (e.g. 5.3) to balls
            overs_int = int(overs)
            balls_int = int(round((overs - overs_int) * 10))
            if balls_int > 5:
                st.error("Invalid over format. The decimal part should represent balls (0 to 5). E.g., 5.3 for 5 overs and 3 balls.")
                return
                
            balls_bowled = (overs_int * 6) + balls_int
            balls_left = 120 - balls_bowled
            wickets_left = 10 - wickets
            
            if balls_bowled == 0:
                crr = 0.0
            else:
                crr = (score * 6) / balls_bowled
                
            if balls_left == 0:
                rrr = 0.0 if runs_left <= 0 else 99.0
            else:
                rrr = (runs_left * 6) / balls_left
                
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'target_runs': [target],
                'crr': [crr],
                'rrr': [rrr]
            })
            
            result = pipe.predict_proba(input_df)
            loss_prob = round(result[0][0] * 100, 1)
            win_prob = round(result[0][1] * 100, 1)
            
            st.markdown(f"<h3 style='text-align: center; color: white;'>Winning Probability</h3>", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Pie(
                labels=[batting_team, bowling_team], 
                values=[win_prob, loss_prob], 
                hole=.5,
                hoverinfo='label+percent',
                textinfo='percent',
                textfont_size=20,
                marker_colors=['#259073', '#e8630a']
            )])
            
            fig.update_layout(
                annotations=[dict(text='Chances', x=0.5, y=0.5, font_size=22, showarrow=False)],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    create_scroll_to_top_button(key_suffix="winPrediction")
