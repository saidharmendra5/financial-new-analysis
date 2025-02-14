import os

model_path = r"C:\Users\sai\OneDrive\Desktop\ST-16-Finsent\models\sentiment_model.pkl"
vectorizer_path = r"C:\Users\sai\OneDrive\Desktop\ST-16-Finsent\models\tfidf_vectorizer.pkl"

print(f"Model Exists: {os.path.exists(model_path)}")
print(f"Vectorizer Exists: {os.path.exists(vectorizer_path)}")



