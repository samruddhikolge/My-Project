import pandas as pd
from sentence_transformers import SentenceTransformer

# Load preprocessed dialogue data
df = pd.read_csv("/home/samruddhi/Project/data/preprocessed_conversations.csv")

# Keep only USER utterances
user_df = df[df["speaker"].str.upper() == "USER"].copy()

utterances = user_df["utterance"].astype(str).tolist()

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(utterances, show_progress_bar=True)

# Create embeddings dataframe
emb_df = pd.DataFrame(
    embeddings,
    columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
)

# Insert utterance column
emb_df.insert(0, "utterance", utterances)

# Save embeddings
output_path = "/home/samruddhi/Project/data/intent_embeddings.csv"
emb_df.to_csv(output_path, index=False)

print("✅ Embeddings regenerated and saved to:", output_path)
print("Shape:", emb_df.shape)
