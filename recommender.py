import streamlit as st
st.set_page_config(page_title="SteamSense", layout="wide")

import os
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql import SparkSession
from config import RAWG_API_KEY

# --- INITIALIZE SPARK ---
spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("SteamRawgRecommender")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

# --- STEAM SCRAPER ---
def get_owned_games_public(steam_id):
    url = f"https://steamcommunity.com/profiles/{steam_id}/games?xml=1"
    resp = requests.get(url); resp.raise_for_status()
    root = ET.fromstring(resp.content)
    owned = {}
    for g in root.findall("games/game"):
        name = g.findtext("name") or ""
        hours = g.findtext("hoursOnRecord") or "0"
        try:
            mins = float(hours) * 60
        except:
            mins = 0
        if name:
            owned[name] = int(mins)
    return owned

# --- RAWG LOADER (via Spark) with checkpoint & terminal progress ---
def load_rawg_spark(api_key, page_size=100, cache_path="rawg_full.parquet"):
    if os.path.exists(cache_path):
        print(f"✔️ Loaded RAWG dataset from cache: {cache_path}")
        return spark.read.parquet(cache_path)

    all_games = []
    page = 1
    print("🚀 Fetching RAWG data...")
    while True:
        resp = requests.get(
            "https://api.rawg.io/api/games",
            params={"key": api_key, "page_size": page_size, "page": page}
        )
        if resp.status_code == 429:
            st.error("Rate limited by RAWG. Try again later.")
            st.stop()
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break

        all_games.extend(batch)
        # checkpoint after each page
        df_checkpoint = pd.DataFrame([{
            "name": g["name"],
            "description": g.get("description_raw","") or g["name"],
            "genres": ";".join(x["name"] for x in g.get("genres",[])),
            "rating": g.get("rating",0),
        } for g in all_games])
        df_checkpoint.to_parquet(cache_path, index=False)
        print(f"✔️ Fetched page {page}: {len(batch)} games; total so far {len(all_games)} saved to {cache_path}")

        page += 1
        time.sleep(0.2)

    print(f"✅ Completed fetching {len(all_games)} games.")
    return spark.createDataFrame(df_checkpoint)

# --- EMBEDDINGS (cached) ---
@st.cache_data(show_spinner=False)
def compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    try:
        model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    except:
        model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    df2 = df.copy()
    df2["genres"] = df2["genres"].fillna("")
    df2["description"] = df2["description"].fillna(df2["name"])
    df2["combined"] = (
        df2["description"] + " " +
        df2["genres"].str.replace(";", " ", regex=False)
    ).str.strip()
    embs = model.encode(df2["combined"].tolist(), show_progress_bar=True, batch_size=16)
    return np.array(embs, dtype="float32")

# --- MAIN APP ---
st.title("🎮 Personalized Steam Game Recommendations")
st.info("Note: Your Steam profile must be **public** to fetch your library.")

if not RAWG_API_KEY:
    st.error("Please set RAWG_API_KEY in config.py.")
    st.stop()

tab1, tab2 = st.tabs(["Owned Games", "Recommendations"])
owned_map = {}

# TAB 1: Load & display Steam library
with tab1:
    steam_id = st.text_input("Enter your SteamID64:")
    if steam_id:
        try:
            owned_map = get_owned_games_public(steam_id)
        except Exception as e:
            st.error(f"Error fetching Steam games: {e}")
    if owned_map:
        df_own = pd.DataFrame([
            {"game": name, "playtime_h": round(mins/60,1)}
            for name, mins in owned_map.items()
        ]).sort_values("playtime_h", ascending=False)
        st.subheader("Your Steam Library")
        st.dataframe(df_own)
    else:
        st.info("Please enter your SteamID64 above to load your library.")

# TAB 2: Recommendations
with tab2:
    if not owned_map:
        st.info("Load your Steam library first.")
    else:
        with st.spinner("Running"):
            rawg_sdf = load_rawg_spark(RAWG_API_KEY)
            df_rawg = rawg_sdf.toPandas()
            embeddings = compute_embeddings(df_rawg)

        # consider only games played >3h
        threshold = 3 * 60
        owned_filtered = {g: m for g, m in owned_map.items() if m >= threshold}
        rawg_names = set(df_rawg["name"])
        owned_valid = {g: m for g, m in owned_filtered.items() if g in rawg_names}

        if not owned_valid:
            st.warning("No owned games over 3 hours found in RAWG dataset.")
        else:
            st.subheader("Matched Owned Games (>3h)")
            st.write(list(owned_valid.keys()))

            # build user profile embedding
            idxs = [i for i, n in enumerate(df_rawg["name"]) if n in owned_valid]
            times = np.array([owned_valid[df_rawg.iloc[i]["name"]] for i in idxs], float)
            weights = times / times.sum()
            user_emb = np.average(embeddings[idxs], axis=0, weights=weights).reshape(1, -1)

            # compute internal sim
            sims = cosine_similarity(user_emb, embeddings).flatten()
            df_rawg["sim"] = sims
            df_rawg.loc[idxs, "sim"] = -1  # exclude owned

            # pick top-rated
            candidates = df_rawg[df_rawg["rating"] >= 3.5]
            recs = candidates.sort_values("sim", ascending=False).head(10)

            disp = recs[["name", "genres", "rating"]].copy()
            st.subheader("🎯 Top 10 High-Rated Game Recommendations")
            st.dataframe(disp.reset_index(drop=True))
