# 🎮 SteamSense

**SteamSense** is a personalized game recommendation engine that uses your public **Steam** library and combines it with the **RAWG** game database to suggest new games you'll likely enjoy. The app leverages **NLP embeddings**, **cosine similarity**, and **Apache Spark** for data processing, all served through a simple **Streamlit UI**.

---

## 🧠 How It Works

1. **Steam Game Fetching**:  
   Enter your **SteamID64** to fetch your publicly owned games and playtime via Steam XML API.

2. **RAWG Game Data**:  
   Game data is fetched (or cached) from the RAWG API, including descriptions, genres, and ratings.

3. **Embeddings with Transformers**:  
   The app generates vector embeddings using `all-mpnet-base-v2` from **SentenceTransformers**, combining game description and genres.

4. **Recommendation Logic**:  
   - Filters owned games with more than 3 hours of playtime.
   - Matches owned games to RAWG dataset.
   - Computes a **user profile embedding** using weighted playtime.
   - Ranks games (not owned) based on **cosine similarity** and **RAWG rating**.
   - Recommends the top 10 high-rated similar games.

---

## 📂 File Structure

```
SteamSense/
├── config.py                  # Stores your RAWG API key
├── rawg_full.parquet          # Cached RAWG dataset for faster loading
├── recommender.py             # Main Streamlit application
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Hamza-chao/SteamSense.git
cd SteamSense
```

### 2. Install requirements

```bash
pip install streamlit pyspark pandas numpy sentence-transformers scikit-learn requests
```

Or generate a `requirements.txt`:

```bash
pip freeze > requirements.txt
```

### 3. Add your API key

Create a file named `config.py`:

```python
# config.py
RAWG_API_KEY = "your_rawg_api_key_here"
```

> 🔐 Your Steam profile **must be public** to fetch owned games.

### 4. Run the app

```bash
streamlit run recommender.py
```

---

## 💡 Features

- ✅ Supports SteamID64 for automatic game fetching
- ✅ RAWG API integration with Spark caching to `rawg_full.parquet`
- ✅ Game vectorization using `SentenceTransformer`
- ✅ Personalized recommendations using cosine similarity
- ✅ Filters games based on rating and playtime
- ✅ Beautiful and responsive UI with Streamlit

---

## 🔍 Example Output

```text
Top 10 High-Rated Game Recommendations:
1. Hades             | Roguelike; Action   | Rating: 4.5
2. Celeste           | Platformer          | Rating: 4.6
3. Risk of Rain 2    | Action; Co-op       | Rating: 4.4
...
```

---

## 📸 Screenshot


```markdown
![App Demo](https://imgur.com/i8YeOiP)
![App Demo](https://imgur.com/i8YeOiP)
```

---

## 📜 License

MIT License © 2025 [Hamza Chao](https://github.com/Hamza-chao)

---

Enjoy your custom-tailored game recommendations with **SteamSense**! 🎮✨
