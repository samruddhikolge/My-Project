import pandas as pd
from collections import defaultdict

# -----------------------------
# Paths
# -----------------------------
clustered_path = "/home/samruddhi/Project/data/phase4_clustered_intents_small.csv"
dialogue_path = "/home/samruddhi/Project/data/preprocessed_conversations.csv"
output_path = "/home/samruddhi/Project/data/cluster_response_bank.csv"

# -----------------------------
# Load data
# -----------------------------
df_clustered = pd.read_csv(clustered_path)
df_dialogue = pd.read_csv(dialogue_path)

print("Loaded clustered data:", df_clustered.shape)
print("Loaded dialogue data:", df_dialogue.shape)

# -----------------------------
# Build response bank
# -----------------------------
rows = []

for _, row in df_clustered.iterrows():
    utterance = row["utterance"]
    cluster_id = row["cluster"]

    # Find matching USER turn
    user_match = df_dialogue[
        (df_dialogue["utterance"] == utterance) &
        (df_dialogue["speaker"].str.upper() == "USER")
    ]

    if user_match.empty:
        continue

    dialogue_id = user_match.iloc[0]["dialogue_id"]
    turn_id = user_match.iloc[0]["turn_id"]
    domain = user_match.iloc[0].get("domain", "unknown")

    # Find next SYSTEM response
    system_turn = df_dialogue[
        (df_dialogue["dialogue_id"] == dialogue_id) &
        (df_dialogue["turn_id"] == turn_id + 1) &
        (df_dialogue["speaker"].str.upper() == "SYSTEM")
    ]

    if not system_turn.empty:
        rows.append({
            "cluster": cluster_id,
            "domain": domain,
            "response": system_turn.iloc[0]["utterance"]
        })

# -----------------------------
# Save response bank
# -----------------------------
df_out = pd.DataFrame(rows).drop_duplicates()
df_out.to_csv(output_path, index=False)

print(" Domain-aware response bank saved to:")
print(output_path)
print(df_out.head())
