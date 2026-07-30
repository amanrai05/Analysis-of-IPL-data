import streamlit as st
import importlib

st.set_page_config(
    page_title="IPL ANALYSIS",
    page_icon="🏏",
    initial_sidebar_state='expanded',
    layout="wide"
)


st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """,
    unsafe_allow_html=True,
)

# Map page names to their module paths (lazy-loaded on navigation)
PAGES = {
    "HOME": "homePage",
    "Predict Score": "scorePrediction",
    "Exploratory Data Analysis": "exploratoryDataAnalysis",
    "Team Analysis": "teamAnalysis",
    "Team v/s Team": "team_vs_teamAnalysis",
    "Batter v/s Bowler": "batter_vs_bowlerAnalysis",
    "Player Analysis": "playerAnalysis",
    "Predict Win Probability": "winnerPrediction",
}

st.sidebar.title('NAVIGATION')
selection = st.sidebar.radio('', list(PAGES.keys()))

# Lazy import: only load the selected page module, not all of them at once.
# This prevents all data-heavy pages from loading datasets at startup.
page = importlib.import_module(PAGES[selection])
page.app()
