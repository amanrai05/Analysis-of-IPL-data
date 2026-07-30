import pandas as pd
import streamlit as st
import os

# -----------------------------------------------------------------------
# Cached data loaders — data is loaded ONCE and cached across reruns.
# 
# To speed up application load times on Render, the computationally
# expensive CSV parsing, string stripping, and memory optimization is now
# performed during the build step by `generate_cache.py`. 
#
# Here, we simply load the pre-computed .pkl files which load in milliseconds
# instead of several seconds.
# -----------------------------------------------------------------------

@st.cache_data(show_spinner="Loading match data...")
def load_matches():
    if os.path.exists('matches_cached.pkl'):
        return pd.read_pickle('matches_cached.pkl')
    else:
        st.error("matches_cached.pkl not found! Please run 'python generate_cache.py' first.")
        return pd.DataFrame()

@st.cache_data(show_spinner="Loading delivery data...")
def load_deliveries():
    if os.path.exists('deliveries_cached.pkl'):
        return pd.read_pickle('deliveries_cached.pkl')
    else:
        st.error("deliveries_cached.pkl not found! Please run 'python generate_cache.py' first.")
        return pd.DataFrame()


# Module-level references for backward compatibility with existing imports.
# Data is read from disk only once and cached by Streamlit across all reruns.
new_matchesDF = load_matches()
new_deliveriesDF = load_deliveries()
