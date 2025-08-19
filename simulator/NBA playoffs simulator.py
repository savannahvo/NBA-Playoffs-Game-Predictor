"""
NBA Playoffs Game Predictor Simulator
=====================================

This Streamlit application simulates the 2025 NBA Playoffs using a trained SVM model.
It provides predictions for playoff games and compares them with actual results,
displaying team statistics, win probabilities, and game outcomes in an interactive interface.

Author: [Savannah Vo]
Date: [August 2025]
Version: 1.0
"""
# Import libraries
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
import datetime
import os
import base64
import re

# Configuration and constants
# Team brand color palettes (primary / secondary / alt)
TEAM_PALETTE_BASE = {
    'Boston Celtics': ['#007A33','#BA9653','#000000'],
    'New York Knicks': ['#006BB6','#F58426','#BEC0C2'],
    'Philadelphia 76ers': ['#006BB6','#ED174C','#002B5C'],
    'Milwaukee Bucks': ['#00471B','#EEE1C6','#000000'],
    'Cleveland Cavaliers': ['#6F263D','#FFB81C','#041E42'],
    'Miami Heat': ['#98002E','#F9A01B','#000000'],
    'Indiana Pacers': ['#002D62','#FDBB30','#BEC0C2'],
    'Orlando Magic': ['#0077C0','#C4CED4','#000000'],
    'Denver Nuggets': ['#0E2240','#FEC524','#8B2131'],
    'Minnesota Timberwolves': ['#0C2340','#78BE20','#236192'],
    'Oklahoma City Thunder': ['#007AC1','#F05A28','#FDBB30'],
    'Phoenix Suns': ['#1D1160','#E56020','#63727A'],
    'Dallas Mavericks': ['#00538C','#8D9093','#002B5E'],
    'Los Angeles Clippers': ['#C8102E','#1D428A','#000000'],
    'Los Angeles Lakers': ['#552583','#FDB927','#000000'],
    'Golden State Warriors': ['#1D428A','#FFC72C','#FF4F00'],
    'Sacramento Kings': ['#5A2D81','#63727A','#000000'],
    'New Orleans Pelicans': ['#0C2340','#85714D','#C8102E'],
    'Memphis Grizzlies': ['#5D76A9','#12173F','#FFD432'],
    'Houston Rockets': ['#CE1141','#000000','#C4CED4'],
}

# Round filter options for sidebar
ROUND_FILTERS = [
    'All',
    'West First Round',
    'East First Round',
    'West Conf. Semifinals',
    'East Conf. Semifinals',
    'West Conf. Finals',
    'East Conf. Finals',
    'NBA Finals',
]

# Round name mapping for extracting round numbers
ROUND_MAP = {
    'First Round': 1,
    'Conference Quarterfinals': 1,
    'Conference Semifinals': 2,
    'Semifinals': 2,
    'Conference Finals': 3,
    'Finals': 4,
    'NBA Finals': 4,
}

# Playoff months for 2025
PLAYOFF_MONTHS_2025 = {4, 5, 6}  

# CSS Styling
def apply_css_styling():
    css = (
        '<style>'
        "@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');"
        ".custom-title {font-family: 'Bebas Neue', sans-serif; font-size: 60px; text-align: center; margin: 10px 0 12px; color: #111;}"
        ".team-name {text-align: center; font-size: 20px; font-weight: 700; margin-top: 5px;}"
        ".series-score {text-align: center; font-size: 14px; color: #555; margin-top: -5px;}"
        ".series-line {text-align:center; font-size: 0.9em; color:#777; margin-top: 6px; margin-bottom: 6px;}"
        ".final-score {text-align: center; font-size: 40px; font-weight: bold; margin-bottom: 5px;}"
        ".logo-wrap {position:relative; display:inline-block;}"
        ".seed-badge {position:absolute; right:-6px; top:-6px; min-width:20px; height:20px; padding:0 6px; border-radius:12px; background:#e9ecef; color:#111; font-weight:700; font-size:12px; display:flex; align-items:center; justify-content:center; border:1px solid rgba(0,0,0,0.1);}"
        ".seed-top {background:#f1c40f; color:#111;}"
        ".seed-mid {background:#dfe3e6; color:#111;}"
        ".seed-rest {background:#e9ecef; color:#111;}"
        ".game-card {background:#fff;border:1px solid #eee;border-radius:16px;padding:18px 16px;margin:18px 0;}"
        ".stat-wrap {display:flex; gap:16px;}"
        ".stat-card {flex:1; background:#fff; border:1px solid #e8e8e8; border-radius:12px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".stat-title {font-weight:700; font-size:16px; margin:0 0 8px;}"
        ".stat-row {display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #f0f0f0; font-size:14px;}"
        ".stat-row:last-child {border-bottom:none;}"
        ".stat-label {color:#666;}"
        ".stat-value {font-weight:700;}"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)

    # Additional CSS for dropdown styling
    st.markdown(
        '''
        <style>
        details summary {
            background-color: white !important;
            color: black !important;
            padding: 6px 12px !important;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
        }
        details[open] summary {
            background-color: white !important;
            color: black !important;
        }
        details summary:hover {
            background-color: #f5f5f5 !important;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

def apply_poster_background():
    # Left poster strip + move content so it doesn't overlap
    POSTER_URL = (
        "https://raw.githubusercontent.com/savannahvo/NBA-Playoffs-Game-Predictor/main/"
        "images/miscellaneous/2025%20NBA%20Playoffs%20Players.jpg"
    )

    st.markdown(
        f"""
        <style>
        :root{{
          --sidebar-w: 240px;   
          --strip-gap: 28px;    
          --strip-w: 360px;     
          --top-gap: 90px;
          --bottom-gap: 140px;
        }}

 
        [data-testid="stSidebar"] {{
          width: var(--sidebar-w) !important;
          min-width: var(--sidebar-w) !important;
          max-width: var(--sidebar-w) !important;
        }}


        .left-poster,
        .left-poster::after {{
          position: fixed;
          pointer-events: none;
          z-index: 0;
        }}

        .left-poster {{
          top: var(--top-gap);
          left: calc(var(--sidebar-w) + var(--strip-gap));
          width: var(--strip-w);
          height: calc(100vh - var(--bottom-gap));
          background-image: url('{POSTER_URL}');
          background-size: auto 100%;
          background-repeat: no-repeat;
          background-position: center top;
          border-radius: 16px;
          box-shadow: 0 12px 28px rgba(0,0,0,.15);
        }}

       
        .left-poster::after {{
          content: "";
          inset: 0;
          border-radius: inherit;
          background: linear-gradient(90deg, rgba(255,255,255,0) 58%, rgba(255,255,255,.96) 100%);
        }}

        .main .block-container {{
          margin-left: calc(var(--strip-gap) + var(--strip-w) + 16px) !important;
          /* remove any padding-left you added earlier */
          padding-left: 0 !important;
          max-width: 100% !important;
          position: relative; z-index: 1;
        }}

    
        @media (max-width: 900px) {{
          .left-poster {{ display:none; }}
          .main .block-container {{ margin-left: 0 !important; }}
        }}
        </style>
        <div class="left-poster"></div>
        """,
        unsafe_allow_html=True
    )


def apply_cinematic_background():
    # Full-page cinematic background 
    BG_URL = (
        "https://raw.githubusercontent.com/savannahvo/NBA-Playoffs-Game-Predictor/main/"
        "images/miscellaneous/2025%20NBA%20Playoffs%20Players.jpg"
    )

    st.markdown(
        f"""
        <style>
    
        #bg-photo {{
          position: fixed;
          inset: 0;
          background-image: url('{BG_URL}');
          background-size: cover;
          background-position: center top;
          background-repeat: no-repeat;
          opacity: 0.34;
          filter: blur(0.6px) saturate(0.85) brightness(0.97);
          z-index: -10; 
        }}

       
        #bg-fade {{
          position: fixed;
          inset: 0;
          pointer-events: none;
          background:
            linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.78) 75%, #ffffff 100%),
            linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.03) 55%, rgba(255,255,255,0.08) 100%);
          z-index: -9;
        }}
        </style>

        <div id="bg-photo"></div>
        <div id="bg-fade"></div>
        """,
        unsafe_allow_html=True
    )



def apply_left_poster_strip():
    # Left poster strip 
    POSTER_URL = (
        "https://raw.githubusercontent.com/savannahvo/NBA-Playoffs-Game-Predictor/main/"
        "images/miscellaneous/2025%20NBA%20Playoffs%20Players.jpg"
    )

    st.markdown(
        f"""
        <style>
        :root {{
          --sidebar-w: 240px;   
          --strip-gap: 28px;    
          --strip-w: 360px;    
          --top-gap: 90px;
          --bottom-gap: 140px;
        }}

        
        [data-testid="stSidebar"] {{
          width: var(--sidebar-w) !important;
          min-width: var(--sidebar-w) !important;
          max-width: var(--sidebar-w) !important;
        }}

       
        .stApp::before,
        .stApp::after {{
          position: fixed;
          top: var(--top-gap);
          left: calc(var(--sidebar-w) + var(--strip-gap));
          width: var(--strip-w);
          height: calc(100vh - var(--bottom-gap));
          border-radius: 16px;
          pointer-events: none;
          z-index: 0; /* below content, above page bg */
          content: "";
        }}

        .stApp::before {{
          background-image: url('{POSTER_URL}');
          background-size: auto 100%;
          background-repeat: no-repeat;
          background-position: center top;
          box-shadow: 0 12px 28px rgba(0,0,0,.15);
        }}

       
        .stApp::after {{
          background: linear-gradient(90deg, rgba(255,255,255,0) 58%, rgba(255,255,255,.96) 100%);
        }}

   
        .main .block-container {{
          position: relative;
          z-index: 1;
          margin-left: calc(var(--strip-gap) + var(--strip-w) + 16px) !important;
          max-width: 100% !important;
          padding-left: 0 !important;  /* ensure no extra padding fights margin */
        }}

     
        @media (max-width: 900px) {{
          .stApp::before, .stApp::after {{ display: none; }}
          .main .block-container {{ margin-left: 0 !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



# Color utility functions
def _hex_to_rgb(h):
    # Convert hex color to RGB tuple
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def _rgb_to_hex(rgb):
    # Convert RGB tuple to hex color
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(v)))) for v in rgb)

def _relative_luminance(rgb):
    # Calculate relative luminance for contrast calculations
    srgb = [v/255 for v in rgb]
    def lin(c): return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
    R,G,B = [lin(c) for c in srgb]
    return 0.2126*R + 0.7152*G + 0.0722

def _contrast_ratio(hex1, hex2):
    # Calculate contrast ratio between two hex colors
    L1 = _relative_luminance(_hex_to_rgb(hex1))
    L2 = _relative_luminance(_hex_to_rgb(hex2))
    L1, L2 = max(L1, L2), min(L1, L2)
    return (L1 + 0.05) / (L2 + 0.05)

def _lightness_tweak(hex_color, factor=1.2):
    # Adjust lightness of a hex color
    r,g,b = _hex_to_rgb(hex_color)
    import colorsys as _cs
    h,l,s = _cs.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, min(1, l*factor))
    rr,gg,bb = _cs.hls_to_rgb(h, l, s)
    return _rgb_to_hex((rr*255, gg*255, bb*255))

def pick_brand_contrasting_colors(home_team, away_team, team_palette):
    # Pick contrasting brand colors for home and away teams
    home_opts = team_palette.get(home_team, ['#d6242c'])
    away_opts = team_palette.get(away_team, ['#17408b'])
    home = home_opts[0]
    best = max(away_opts, key=lambda c: _contrast_ratio(home, c))
    cr = _contrast_ratio(home, best)
    if cr < 2.5:
        for f in (1.25, 1.4, 0.8, 0.65):
            tweaked = _lightness_tweak(best, f)
            new_cr = _contrast_ratio(home, tweaked)
            if new_cr > cr:
                best, cr = tweaked, new_cr
            if cr >= 2.5:
                break
    return home, best

# Data processing helper functions
def round_matches_label(game_label, selected_label):
    # Check if game label matches selected round filter
    if selected_label == 'All':
        return True
    if not isinstance(game_label, str):
        return False
    gl = game_label.lower()
    return selected_label.lower() in gl

def extract_round_number(game_label):
    # Extract round number from game label
    if not isinstance(game_label, str):
        return None
    gl = game_label.lower()
    for k, v in ROUND_MAP.items():
        if k.lower() in gl:
            return v
    if 'first round' in gl: return 1
    if 'semifinal' in gl: return 2
    if 'conference finals' in gl: return 3
    if 'finals' in gl: return 4
    return None

def extract_game_number(game_sub_label):
    # Extract game number from game sub-label
    if not isinstance(game_sub_label, str):
        return None
    m = re.search(r'Game\s+(\d+)', game_sub_label, re.IGNORECASE)
    return int(m.group(1)) if m else None

def get_series_record(df, team1, team2, current_date, current_game_id=None):
    # Get series record between two teams up to current date
    prior_games = df[
        (((df['homeTeam'] == team1) & (df['awayTeam'] == team2)) |
         ((df['homeTeam'] == team2) & (df['awayTeam'] == team1))) &
        (df['gameDate'] < current_date)
    ]
    if current_game_id:
        prior_games = prior_games[prior_games['gameId'] != current_game_id]
    t1, t2 = 0, 0
    for _, g in prior_games.iterrows():
        if pd.notnull(g['homeScore']) and pd.notnull(g['awayScore']):
            winner = g['homeTeam'] if g['homeScore'] > g['awayScore'] else g['awayTeam']
            if winner == team1: t1 += 1
            elif winner == team2: t2 += 1
    return t1, t2

# File and image utility functions
def load_logo(team):
    # Load team logo as base64 encoded string
    path = f'images/team logos/{team}.png'
    if os.path.exists(path):
        with open(path, 'rb') as img:
            return base64.b64encode(img.read()).decode()
    return ''

def load_image_b64(path: str) -> str:
    # Load image file as base64 encoded string
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# Model utility functions
def get_feature_cols_from_model(model):
    # Extract feature column names from trained model
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for _, step in model.named_steps.items():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

# Formatting utility functions
def fmt_pct(val):
    # Format value as percentage
    try:
        v = float(val)
    except Exception:
        return 'N/A'
    if v <= 1.0:
        v *= 100.0
    return f'{v:.1f}%'

def fmt_ratio(val):
    # Format value as ratio
    try:
        v = float(val)
    except Exception:
        return 'N/A'
    if v > 3.0:
        v /= 100.0
    return f'{v:.3f}'

def colored_value(metric, my_val, opp_val):
    # Return colored value based on comparison with opponent
    def to_float(x):
        try: return float(x)
        except: return None
    my = to_float(my_val)
    op = to_float(opp_val)
    if my is None:
        return ('N/A', '#444')
    higher_better = metric in ('eFG%', 'ORB%', 'DRB%', 'FT/FGA')
    if metric == 'TOV%':
        higher_better = False
    def norm(x, kind):
        if x is None: return None
        if kind in ('eFG%', 'TOV%', 'ORB%', 'DRB%'):
            return x*100.0 if x <= 1.0 else x
        if kind == 'FT/FGA':
            return x/100.0 if x > 3.0 else x
        return x
    my_n = norm(my, metric)
    op_n = norm(op, metric) if op is not None else None
    color = '#444'
    if op_n is not None:
        if higher_better:
            color = '#1a7f37' if my_n > op_n else ('#c92a2a' if my_n < op_n else '#444')
        else:
            color = '#1a7f37' if my_n < op_n else ('#c92a2a' if my_n > op_n else '#444')
    text = fmt_ratio(my) if metric == 'FT/FGA' else fmt_pct(my)
    return (text, color)

# UI Rendering Functions
def render_stats_card(team_name, s, opp_s):
    # Render team statistics card HTML
    rows = []
    for m in ['eFG%', 'TOV%', 'ORB%', 'DRB%', 'FT/FGA']:
        val, color = colored_value(m, s.get(m, 'N/A'), opp_s.get(m, 'N/A') if opp_s is not None else None)
        rows.append((m, f"<span style='color:{color}'>{val}</span>"))
    html = (
        "<div class='stat-card'>"
        f"<div class='stat-title'>{team_name}</div>"
        + "".join([
            f"<div class='stat-row'><span class='stat-label'>{k}</span><span class='stat-value'>{v}</span></div>"
            for k, v in rows
        ])
        + "</div>"
    )
    return html

def render_explanation_centered():
    # Render centered explanation section with accordion
    st.markdown(
        '''
        <div style="max-width: 820px; margin: 0 auto; text-align: center;">
          <p style="font-size:16px; color:#555; margin: 8px 0 10px;">
            This simulator covers the 2025 NBA Playoffs. It compares a support vector machine (SVM) model's
            predicted game winner and win probability, shown in the prediction bar, with the real box score and
            team stats from games that already happened. Each playoff round and game was modeled separately, with
            different features depending on round and game number.
          </p>

          <details style="display:inline-block; margin: 6px 0 12px; text-align:left;">
            <summary style="list-style:none; cursor:pointer; display:inline-flex; align-items:center;
                             gap:8px; color:#444; font-weight:600; border:1px solid #cfe0ff; padding:6px 10px;
                             border-radius:8px; background:#f5f8ff;">
              <span style="display:inline-flex; width:18px; height:18px; border-radius:50%;
                           background:#eef3ff; color:#1d4ed8; align-items:center; justify-content:center;
                           font-size:12px; border:1px solid #d9e2ff;">i</span>
              <span>How the Simulator Works</span>
            </summary>
            <div style="padding:10px 2px 0; color:#555; font-size:14px; line-height:1.55;">
              <ul style="margin:0; padding-left:18px;">
                <li><b>Model and Data:</b> SVM classification trained on 2015–2022 playoffs, validated on 2023–2024, tested on 2025 playoffs.</li>
                <li><b>Why Start in 2015?</b> League-wide three-point attempts per game rose from about 22.4 in 2013–14 to 26.8 in 2014–15, then continued climbing, along with higher effective field goal percentage. This reflects a shift to modern spacing and higher-scoring games, so modeling from 2015 captures the current era.</li>
                <li><b>Filters:</b> Use the sidebar to filter by date or date range, by round (West/East Rounds or NBA Finals).</li>
                <li><b>What You See:</b> The prediction bar is the model's output; the final score and team stats are actual game results. Use them together to judge accuracy.</li>
                <li><b>Feature Design:</b> Feature sets differ by round and game number to reflect changing context across a series (for example, regular season baselines for early games, rolling playoff form and series state for later games).</li>
              </ul>
            </div>
          </details>
        </div>
        ''',
        unsafe_allow_html=True
    )

def seed_class(n):
    # Return CSS class for seed badge based on seed number
    if n is None: return 'seed-rest'
    if n <= 2: return 'seed-top'
    if n <= 4: return 'seed-mid'
    return 'seed-rest'

# Data loading functions
@st.cache_data
def load_data():
    # Load game data and actual statistics (cached for performance)
    g_path = 'data/final/simulator_data.csv'
    a_path = 'data/processed/team_statistics_playoff_games.csv'

    if not os.path.exists(g_path):
        raise FileNotFoundError(f"Missing file: {g_path}")
    if not os.path.exists(a_path):
        raise FileNotFoundError(f"Missing file: {a_path}")

    g = pd.read_csv(g_path, parse_dates=['gameDate'])
    a = pd.read_csv(a_path)
    return g, a

# Main application function
st.set_page_config(layout="wide")
def main():
    # Main Streamlit application function
    
    # Apply all styling
    apply_css_styling()
    apply_poster_background()
    apply_cinematic_background()
    apply_left_poster_strip()
    
    st.markdown('''
        <style>
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("images/miscellaneous/2025 NBA Playoffs Players.jpg");
            background-size: cover;
            background-position: center;
            opacity: 0.15; 
            z-index: -1; 
        }
        </style>
    ''', unsafe_allow_html=True)

    # Hero image and title
    st.image('images/miscellaneous/2025 NBA Playoffs Court.jpeg', use_container_width=True)
    st.markdown('<div class="custom-title">2025 NBA PLAYOFFS GAME PREDICTOR</div>', unsafe_allow_html=True)
    render_explanation_centered()

    # Load model and data
    model = joblib.load("model/final_pipeline_model.pkl")
    all_games, actual_stats = load_data()

    # Add round and game number columns if not present
    if 'roundNumber' not in all_games.columns or 'gameNumber' not in all_games.columns:
        all_games['roundNumber'] = all_games['gameLabel'].apply(extract_round_number)
        all_games['gameNumber'] = all_games['gameSubLabel'].apply(extract_game_number)

    # Get playoff teams and set up team palette
    playoff_teams = sorted(set(all_games['homeTeam']).union(set(all_games['awayTeam'])))
    TEAM_PALETTE = {t: TEAM_PALETTE_BASE.get(t, ['#555555','#C4CED4','#000000']) for t in playoff_teams}

    # Sidebar filters
    st.sidebar.header('Filters')

    # Date filtering
    available_dates = sorted({
        d.date()
        for d in all_games['gameDate']
        if d.year == 2025 and d.month in PLAYOFF_MONTHS_2025
    })

    if not available_dates:
        st.sidebar.warning("No 2025 playoff dates found in data.")
        start_date = end_date = None
        use_date = use_range = False
    else:
        use_date = st.sidebar.checkbox('Filter by date', value=False)
        use_range = st.sidebar.checkbox('Use date range', value=False) if use_date else False

        default_single = available_dates[0]
        default_range = (available_dates[0], available_dates[-1])

        if use_date and use_range:
            start_date, end_date = st.sidebar.select_slider(
                'Date range',
                options=available_dates,
                value=default_range
            )
        elif use_date:
            selected_date = st.sidebar.selectbox(
                'Playoff date',
                options=available_dates,
                index=0
            )
            start_date, end_date = selected_date, selected_date
        else:
            start_date, end_date = None, None

    # Other filters
    round_choice = st.sidebar.selectbox('Round', options=ROUND_FILTERS, index=0)
    game_choice = st.sidebar.selectbox('Game #', options=['All'] + list(range(1, 8)), index=0)
    team_multi = st.sidebar.multiselect('Teams', playoff_teams, default=[])
    use_cb = st.sidebar.toggle('Color-blind mode', value=False)

    
    # Apply filters to data
    filtered = all_games.copy()
    
    # Date filter
    if use_date and start_date is not None and end_date is not None:
        d0, d1 = min(start_date, end_date), max(start_date, end_date)
        mask = (filtered['gameDate'].dt.date >= d0) & (filtered['gameDate'].dt.date <= d1)
        filtered = filtered[mask]
    
    # Round filter
    if round_choice != 'All':
        filtered = filtered[filtered['gameLabel'].apply(lambda x: round_matches_label(x, round_choice))]
    
    # Game number filter
    if game_choice != 'All':
        filtered = filtered[filtered['gameNumber'] == int(game_choice)]
    
    # Team filter
    if team_multi:
        filtered = filtered[(filtered['homeTeam'].isin(team_multi)) | (filtered['awayTeam'].isin(team_multi))]

    if filtered.empty:
        st.info('No games match the selected filters.')
        return

    # Prepare features for model prediction
    # Define non-feature columns to exclude from model input
    non_feature_cols = [
        'gameDate', 'true_label', 'predicted_label', 'predicted_winner', 'actual_winner',
        'awayTeam', 'homeTeam',
        'homeWin', 'awayWin',
        'homeScore', 'awayScore',
        'gameLabel', 'gameSubLabel',
        'roundNumber'
    ]

    # Load saved feature list
    feature_list_path = os.path.join('model/', 'final_svm_model_features.txt')
    saved_feature_cols = None
    if os.path.exists(feature_list_path):
        with open(feature_list_path, 'r') as f:
            saved_feature_cols = [line.strip() for line in f if line.strip()]

    # Get expected model features
    model_expected_cols = saved_feature_cols or get_feature_cols_from_model(model)
    
    # Clean up filtered data
    filtered = filtered.drop(columns=['seriesId', 'matchupType', 'conference_home', 'conference_away'], errors='ignore')

    # Prepare feature matrix
    X_df = filtered.drop(columns=non_feature_cols, errors='ignore').copy()

    # Align features with model expectations
    if model_expected_cols is not None:
        missing_cols = [c for c in model_expected_cols if c not in X_df.columns]
        extra_cols = [c for c in X_df.columns if c not in model_expected_cols]
        for c in missing_cols:
            X_df[c] = float('nan')
        if extra_cols:
            X_df = X_df.drop(columns=extra_cols, errors='ignore')
        X_df = X_df.reindex(columns=model_expected_cols)
        if missing_cols or extra_cols:
            st.warning(f'Adjusted feature columns to match the model. Missing added as NaN: {missing_cols} | Dropped extras: {extra_cols}')
    else:
        st.info('Model feature list not found; consider saving ../model/final_svm_model_features.txt during training.')

    # Generate model predictions
    try:
        preds = model.predict(X_df)
        probs = model.predict_proba(X_df)
    except Exception as e:
        st.error(f'Prediction failed: {e}')
        st.stop()

    # Add predictions to filtered data
    filtered = filtered.copy()
    filtered['Predicted Winner'] = preds
    filtered['Win Probability'] = probs.max(axis=1)

    # Render game cards
    for _, row in filtered.iterrows():
        st.markdown("<div class='game-card'>", unsafe_allow_html=True)

        # Extract team information
        home_team = row['homeTeam']
        away_team = row['awayTeam']

        # Extract and process seed information
        home_seed = row.get('homeSeed', None)
        away_seed = row.get('awaySeed', None)
        try: home_seed = int(home_seed) if pd.notnull(home_seed) else None
        except: home_seed = None
        try: away_seed = int(away_seed) if pd.notnull(away_seed) else None
        except: away_seed = None

        # Get series record and game information
        home_series, away_series = get_series_record(all_games, home_team, away_team, row['gameDate'], row['gameId'])
        game_label = str(row.get('gameLabel', '')).strip()
        game_sub_label = str(row.get('gameSubLabel', '')).strip()
        game_date_str = row['gameDate'].strftime('%b %d')

        # Render game header
        st.markdown(f"<div class='series-score'>NBA · {game_date_str}</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 4, 3])

        # Render home team (left column)
        with col1:
            left_html = (
                "<div style='display:flex;flex-direction:column;align-items:center;'>"
                "<div class='logo-wrap'>"
                f"<img src='data:image/png;base64,{load_logo(home_team)}' width='90'/>"
                + (f"<div class='seed-badge {seed_class(home_seed)}' title='Conference seed'>{home_seed}</div>" if home_seed is not None else "")
                + "</div>"
                f"<div class='team-name'>{home_team}</div>"
                f"<div class='series-score'>({home_series} - {away_series})</div>"
                "</div>"
            )
            st.markdown(left_html, unsafe_allow_html=True)

        # Render final score (center column)
        with col2:
            st.markdown(f"<div class='final-score'>{int(row['homeScore'])} - {int(row['awayScore'])}</div>", unsafe_allow_html=True)
            st.markdown("<div class='series-score'>Final</div>", unsafe_allow_html=True)

        # Render away team (right column)
        with col3:
            right_html = (
                "<div style='display:flex;flex-direction:column;align-items:center;'>"
                "<div class='logo-wrap'>"
                f"<img src='data:image/png;base64,{load_logo(away_team)}' width='90'/>"
                + (f"<div class='seed-badge {seed_class(away_seed)}' title='Conference seed'>{away_seed}</div>" if away_seed is not None else "")
                + "</div>"
                f"<div class='team-name'>{away_team}</div>"
                f"<div class='series-score'>({away_series} - {home_series})</div>"
                "</div>"
            )
            st.markdown(right_html, unsafe_allow_html=True)

        # Render series information line
        series_suffix = ""
        game_num = row.get('gameNumber')
        if pd.notnull(game_num) and int(game_num) > 1:
            if home_series > away_series:
                series_suffix = f"{home_team} leads {home_series} - {away_series}"
            elif away_series > home_series:
                series_suffix = f"{away_team} leads {away_series} - {home_series}"
            else:
                series_suffix = f"Series tied {home_series} - {away_series}"
        series_line = f"{game_label} · {game_sub_label}" + (f" · {series_suffix}" if series_suffix else "")
        st.markdown(f"<div class='series-line'>{series_line}</div>", unsafe_allow_html=True)

        # Render prediction bar
        # Determine predicted winner and probabilities
        winner_is_home = bool(row['Predicted Winner'])
        pred_team = home_team if winner_is_home else away_team
        home_prob = row['Win Probability'] if winner_is_home else 1 - row['Win Probability']
        away_prob = 1 - home_prob

        # Choose colors based on color-blind mode setting
        if use_cb:
            # Color-blind friendly palette
            cb = ['#0072B2','#E69F00','#56B4E9','#D55E00','#F0E442','#009E73','#CC79A7','#999999']
            home_color = cb[hash(home_team) % len(cb)]
            away_color = cb[hash(away_team) % len(cb)]
        else:
            # Team brand colors with contrast optimization
            home_color, away_color = pick_brand_contrasting_colors(home_team, away_team, TEAM_PALETTE)

        # Render prediction information
        st.markdown(f"<div class='series-score'><b>Predicted Winner:</b> {pred_team}</div>", unsafe_allow_html=True)

        # Render probability bar
        bar_html = (
            "<div style='display:flex;flex-direction:column;align-items:center;margin-top:6px;margin-bottom:15px;'>"
            "<div style='width:80%;display:flex;font-weight:bold;font-size:14px;justify-content:space-between;margin-bottom:5px;'>"
            f"<span>{home_team} ({home_prob:.2%})</span>"
            f"<span>{away_team} ({away_prob:.2%})</span>"
            "</div>"
            "<div style='width:80%;height:20px;background:#eee;border-radius:10px;overflow:hidden;display:flex;position:relative;'>"
            f"<div style='width:{home_prob*100:.2f}%;background:{home_color};'></div>"
            f"<div style='width:{away_prob*100:.2f}%;background:{away_color};'></div>"
            f"<div style='position:absolute;left:{home_prob*100:.2f}%;top:0;bottom:0;width:1px;background:#fff;'></div>"
            "</div>"
            "</div>"
        )
        st.markdown(bar_html, unsafe_allow_html=True)

        # Render team statistics
        with st.expander('Team Stats'):
            # Get actual statistics for both teams
            actual_home = actual_stats[
                (actual_stats['gameId'] == row['gameId']) &
                (actual_stats['teamName'] == home_team)
            ].squeeze()
            actual_away = actual_stats[
                (actual_stats['gameId'] == row['gameId']) &
                (actual_stats['teamName'] == away_team)
            ].squeeze()
            
            # Render statistics cards in two columns
            c1, c2 = st.columns(2)
            c1.markdown(render_stats_card(home_team, actual_home, actual_away), unsafe_allow_html=True)
            c2.markdown(render_stats_card(away_team, actual_away, actual_home), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # end game-card

# APPLICATION ENTRY POINT
if __name__ == '__main__':
    main()