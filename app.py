import os
import re
import pandas as pd
import numpy as np
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# FLASK KURULUM
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# ALERJEN TANIMLAMALARI
# ─────────────────────────────────────────────
ALLERGEN_KEYWORDS = {
    "Gluten": [
        "wheat", "flour", "bread", "pasta", "noodle", "bun", "dough",
        "barley", "rye", "semolina", "crouton", "breaded", "battered",
        "soy sauce", "teriyaki", "sandwich", "wrap", "pita", "tortilla",
        "roll", "toast", "bagel", "beer", "beers", "ale", "malt","ipa", "esb", "lager", 
        "stout", "porter", "pilsner", "wheat beer", "brewing", "brewery","spaghetti", "spaghettini", 
        "macaroni", "fettuccine", "fettucinne", "penne", "ravioli", "linguine", "lasagna",
    ],
    "Dairy": [
        "milk", "cheese", "cream", "butter", "yogurt", "mozzarella",
        "parmesan", "brie", "cheddar", "ricotta", "whey", "lactose",
        "bechamel", "alfredo", "latte", "cappuccino", "macchiato"
    ],
    "Egg": [
        "egg", "eggs", "omelette", "omelet","frittata", "mayonnaise", "mayo",
        "hollandaise", "meringue", "custard"
    ],
    "Nuts": [
        "almond", "walnut", "cashew", "pistachio", "hazelnut", "pecan",
        "macadamia", "pine nut", "nut", "praline", "marzipan"
    ],
    "Peanut": ["peanut", "peanut butter", "groundnut", "satay"],

    "Shellfish": [
        "shrimp", "prawn", "lobster", "crab", "crayfish", "scallop",
        "clam", "oyster", "mussel", "squid", "octopus","seafood"
    ],
    "Fish": [
        "salmon", "tuna", "cod", "bass", "anchovy", "sardine", "tilapia",
        "halibut", "trout", "mackerel", "fish", "swordfish","catfish",
        "milkfish", "bangus", "herring", "snapper", "flounder", "perch",
        "pike", "carp", "sole", "turbot","seafood"
    ],
    "Soy": [
        "soy", "soya", "tofu", "edamame", "miso", "tempeh",
        "soy sauce", "tamari"
    ],
    "Sesame": ["sesame", "tahini", "hummus", "halva"],

    "Sulfite": [
        "wine", "vinegar", "dried fruit", "raisin", "sultana",
        "pickled", "preserved"
    ]
}

# ─────────────────────────────────────────────
# ALERJEN FONKSİYONLARI
# ─────────────────────────────────────────────
def detect_allergens(text: str) -> list:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    text_lower = text.lower()
    found = []
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r's?\b'
            if re.search(pattern, text_lower):
                found.append(allergen)
                break
    return found

def add_allergen_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["allergens"] = (
        df["name"].fillna("") + " " + df["description"].fillna("")
    ).apply(detect_allergens)
    df["allergen_str"] = df["allergens"].apply(
        lambda x: ", ".join(x) if x else "None"
    )
    return df
def filter_by_allergen(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    if "allergens" not in df.columns:
        df = add_allergen_column(df)
    exclude_set = set(exclude)
    return df[df["allergens"].apply(
        lambda a: len(set(a) & exclude_set) == 0
    )]

# ─────────────────────────────────────────────
# VERİ SETİ YÜKLEME VE TEMİZLEME
# CSV DOSYASININ YOLUNU KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN
# ─────────────────────────────────────────────
print("Veri seti yükleniyor...")
df_big = pd.read_csv(r"C:\Users\nefise.durmaz\Desktop\Bitirme Çalışması\food_menu_items.csv")
df_big = df_big.drop('menu_url', axis=1)
df_big = df_big.replace("nan", np.nan)

giysi_keywords = ["pullover", "t-shirt", "shirt", "jacket", "pants", "dress", "sweater", "hoodie"]
mask = df_big["name"].str.lower().str.contains("|".join(giysi_keywords), na=False)
df_big = df_big[~mask].reset_index(drop=True)

df_big["name"] = df_big["name"].str.replace(r"^[#\d\s]*\.?\s*", "", regex=True).str.strip()
df_big = df_big.drop_duplicates().reset_index(drop=True)
df_big = df_big.drop_duplicates(subset=["name", "restaurant_name"]).reset_index(drop=True)
df_big = df_big.dropna(subset=["name", "description"])
df_big = df_big[df_big["name"].str.len() >= 3]
df_big = df_big[~df_big["name"].str.match(r"^\d+$", na=False)]
df_big = df_big[df_big["description"].str.strip().str.len() >= 10]
df_big = df_big[df_big["description"].str.split().str.len() > 1]
df_big = df_big.reset_index(drop=True)
print(f"Veri seti hazır: {len(df_big)} satır")
print("Alerjenler hesaplanıyor...")
df_big = add_allergen_column(df_big)
print("Alerjenler hazır.")

# ─────────────────────────────────────────────
# BM25 İNDEKSİ
# ─────────────────────────────────────────────
print("BM25 indeksi oluşturuluyor...")
df_big["search_text"] = (
    df_big["name"].fillna("").astype(str) + " " +
    df_big["description"].fillna("").astype(str) + " " +
    df_big["category"].fillna("").astype(str) + " " +
    df_big["cuisine"].fillna("").astype(str)
)
tokenized_corpus = [text.lower().split() for text in df_big["search_text"]]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 hazır.")

def bm25_search(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[::-1][:top_k]
    results = df_big.iloc[top_indices].copy()
    results["bm25_score"] = scores[top_indices]
    return results
def embedding_search(query, top_k=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, text_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    results = df_big.iloc[top_indices].copy()
    results["semantic_score"] = similarities[top_indices] * 100
    return results

def hybrid_search(query, top_k=10, bm25_weight=0.5, embedding_weight=0.5):
    k = 60  # RRF sabit parametresi
    
    # Her iki yöntemden sonuçları al
    bm25_results = bm25_search(query, top_k=top_k*3)
    emb_results = embedding_search(query, top_k=top_k*3)
    
    # RRF skorlarını hesapla
    rrf_scores = {}
    
    # BM25 sıraları
    for rank, (_, row) in enumerate(bm25_results.iterrows()):
        name = row['name']
        restaurant = row['restaurant_name']
        key = f"{name}||{restaurant}"
        if key not in rrf_scores:
            rrf_scores[key] = {'score': 0, 'row': row}
        rrf_scores[key]['score'] += bm25_weight * (1 / (k + rank + 1))
    
    # Embedding sıraları
    for rank, (_, row) in enumerate(emb_results.iterrows()):
        name = row['name']
        restaurant = row['restaurant_name']
        key = f"{name}||{restaurant}"
        if key not in rrf_scores:
            rrf_scores[key] = {'score': 0, 'row': row}
        rrf_scores[key]['score'] += embedding_weight * (1 / (k + rank + 1))
    
    # Skora göre sırala
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    
    # DataFrame oluştur
    rows = [item['row'] for item in sorted_results]
    scores = [item['score'] for item in sorted_results]
    
    result_df = pd.DataFrame(rows).reset_index(drop=True)
    result_df['hybrid_score'] = scores
    
    return result_df
def hybrid_search_on(query, dataframe, top_k=10, bm25_weight=0.5, embedding_weight=0.5):
    k = 60
    
    # Filtrelenmiş df'nin indexlerini al
    filtered_indices = dataframe.index.tolist()
    
    # Ana embedding'lerden sadece filtrelenmiş satırları al
    filtered_embeddings = text_embeddings[filtered_indices]
    
    # BM25 için geçici indeks
    temp_search_text = (
        dataframe["name"].fillna("").astype(str) + " " +
        dataframe["description"].fillna("").astype(str) + " " +
        dataframe["category"].fillna("").astype(str) + " " +
        dataframe["cuisine"].fillna("").astype(str)
    )
    temp_corpus = [text.lower().split() for text in temp_search_text]
    temp_bm25 = BM25Okapi(temp_corpus)

    # BM25 arama
    tokenized_query = query.lower().split()
    bm25_scores = temp_bm25.get_scores(tokenized_query)  # ← bu eksik

    search_size = min(500, len(dataframe))
    bm25_top = bm25_scores.argsort()[::-1][:search_size]

    # Embedding arama - önceden hesaplanmış embedding'leri kullan
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
    emb_top = similarities.argsort()[::-1][:search_size]
    # RRF birleştirme
    rrf_scores = {}
    for rank, idx in enumerate(bm25_top):
        row = dataframe.iloc[idx]
        key = f"{row['name']}||{row['restaurant_name']}"
        if key not in rrf_scores:
            rrf_scores[key] = {'score': 0, 'row': row}
        rrf_scores[key]['score'] += bm25_weight * (1 / (k + rank + 1))

    for rank, idx in enumerate(emb_top):
        row = dataframe.iloc[idx]
        key = f"{row['name']}||{row['restaurant_name']}"
        if key not in rrf_scores:
            rrf_scores[key] = {'score': 0, 'row': row}
        rrf_scores[key]['score'] += embedding_weight * (1 / (k + rank + 1))

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    result_df = pd.DataFrame([item['row'] for item in sorted_results]).reset_index(drop=True)
    result_df['hybrid_score'] = [item['score'] for item in sorted_results]

    return result_df
# ─────────────────────────────────────────────
# EMBEDDING MODELİ
# ─────────────────────────────────────────────
print("Embedding modeli yükleniyor...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = (df_big["name"].astype(str) + " - " + df_big["description"].astype(str)).tolist()

EMBEDDING_PATH = r"C:\Users\nefise.durmaz\Desktop\Bitirme Çalışması\embeddings.npy"

if os.path.exists(EMBEDDING_PATH):
    print("Kaydedilmiş embedding'ler yükleniyor...")
    text_embeddings = np.load(EMBEDDING_PATH)
    print("Embedding yüklendi.")
else:
    print("Embedding'ler hesaplanıyor, bu biraz sürebilir...")
    text_embeddings = model.encode(texts, show_progress_bar=True)
    np.save(EMBEDDING_PATH, text_embeddings)
    print("Embedding hesaplandı ve kaydedildi.")

# ─────────────────────────────────────────────
# OLLAMA SİSTEM TALİMATI
# ─────────────────────────────────────────────
system_instruction = (
    "You are a restaurant assistant. "
    "You have ONLY these products available: the ones listed under 'Menu Information'. "
    "Do NOT mention, invent, or suggest ANY product that is not in this exact list. "
    "When recommending a product, ALWAYS mention the restaurant name where it is available. "
    "Recommend ONLY from the listed products by their exact names. "
    "Carefully read each product's description before recommending it. "
    "If the customer says they dislike or want to avoid an ingredient, "
    "do NOT recommend any product whose description mentions that ingredient. "
    "Keep response under 3 sentences."
)

# ─────────────────────────────────────────────
# FLASK API ENDPOİNT'LERİ
# ─────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    user_allergens = data.get('allergens', [])
    history = data.get('history', [])  # ← ekleyin

    if not query:
        return jsonify({'error': 'Sorgu boş'}), 400

    # Alerjen varsa önce veri setini filtrele
    if user_allergens:
        filtered_df = filter_by_allergen(df_big, exclude=user_allergens)
    else:
        filtered_df = df_big

    city = data.get('city', None)    
    if city:
        city_filtered = filtered_df[filtered_df["city_name"].str.lower() == city.lower()]
        if len(city_filtered) > 0:
            filtered_df = city_filtered
    # Filtrelenmiş veri seti üzerinde arama yap
    results = hybrid_search_on(query, filtered_df, top_k=50)
    results = add_allergen_column(results)

    # Alerjen uyarıları için orijinal veri setinde arama
    warnings = []
    if user_allergens:
        warning_results = hybrid_search(query, top_k=50)
        warning_results = add_allergen_column(warning_results)
        for _, row in warning_results.iterrows():
            matches = set(row["allergens"]) & set(user_allergens)
            if matches:
                warnings.append(f"{row['name']} — {', '.join(matches)}")

    # Context oluştur
    context_text = ""
    safe_count = 0
    for _, row in results.iterrows():
        desc = str(row['description'])
        if safe_count < 10 and len(desc.split()) >= 4:
            context_text += f"- Product: {row['name']} | Restaurant: {str(row.get('restaurant_name', ''))} | City: {str(row.get('city_name', ''))} | Description: {desc}\n"
            safe_count += 1

    print(f"Context text:\n{context_text}") 

    if not context_text:
        return jsonify({
            'answer': "I'm sorry, I couldn't find any suitable options for your request.",
            'warnings': warnings
        })

    response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {'role': 'system', 'content': system_instruction},
        *history[:-1],  # Son mesaj hariç geçmiş
        {'role': 'user', 'content': f"Menu Information:\n{context_text}\n\nCustomer's Question: {query}"}
    ]
)

    return jsonify({
        'answer': response['message']['content'],
        'warnings': warnings
    })
@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'hybrid')
    top_k = data.get('top_k', 10)
    city = data.get('city', None)

    if not query:
        return jsonify({'error': 'Sorgu boş'}), 400

    search_pool = top_k * 10  # daha fazla sonuç al, sonra şehre göre filtrele

    if method == 'bm25':
        results = bm25_search(query, top_k=search_pool)
        results = results.rename(columns={"bm25_score": "score"})
    elif method == 'hybrid':
        results = hybrid_search(query, top_k=search_pool)
        results = results.rename(columns={"hybrid_score": "score"})
    else:
        results = embedding_search(query, top_k=search_pool)
        results = results.rename(columns={"semantic_score": "score"})

    # Şehre göre filtrele
    if city:
        city_results = results[results["city_name"].str.lower() == city.lower()]
        if len(city_results) > 0:
            results = city_results

    results = results.head(top_k)

    return jsonify(results[["name", "category", "restaurant_name", "city_name"]].to_dict(orient='records'))
# ─────────────────────────────────────────────
# UYGULAMAYI BAŞLAT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Flask API başlatılıyor: http://localhost:5000")
    app.run(port=5000, debug=False)
