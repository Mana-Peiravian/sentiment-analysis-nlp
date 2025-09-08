import json, os
import joblib
import numpy as np

os.makedirs("app/model", exist_ok=True)

clf = joblib.load("models/logreg_model.pkl")       # LogisticRegression
vec = joblib.load("models/tfidf.pkl")              # TfidfVectorizer

# Build a dense vector of idf aligned to feature indices 0..(n_features-1)
# and a compact vocab map token -> index
vocab_items = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
tokens = [t for t, _ in vocab_items]
indices = [i for _, i in vocab_items]

idf_full = np.zeros(len(indices), dtype=float)
idf_full[:] = np.nan
idf_full[:] = np.array(vec.idf_)[np.argsort(indices)]

model_json = {
    "classes": list(clf.classes_),                       # e.g. ["negative","neutral","positive"]
    "coef": clf.coef_.tolist(),                          # shape [n_classes, n_features]
    "intercept": clf.intercept_.tolist(),                # shape [n_classes]
    "vocabulary": tokens,                                # token list aligned with feature index
    "idf": idf_full.tolist(),                            # idf aligned to same indices
    "meta": {
        "analyzer": "word",
        "lowercase": True,
        "token_pattern": r"(?u)\\b\\w\\w+\\b",
        "use_idf": True,
        "smooth_idf": True,
        "norm": "l2",
        "sublinear_tf": False,
        "n_features": len(tokens)
    }
}

with open("app/model/model.json", "w", encoding="utf-8") as f:
    json.dump(model_json, f, ensure_ascii=False)
print("✅ Wrote app/model/model.json")
