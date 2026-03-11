#Load preprocessed data
import pandas as pd

# Path to your preprocessed file
input_csv = '/home/samruddhi/Project/data/preprocessed_conversations.csv'
df = pd.read_csv(input_csv)

print("Data loaded:", df.shape)
print(df.head(3))
#(A)SEMANTIC FEATURES
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load model (takes a few seconds)
model = SentenceTransformer('all-mpnet-base-v2')

# Compute sentence embeddings (768D)
embeddings = model.encode(df['utterance'].tolist(), show_progress_bar=True)
semantic_df = pd.DataFrame(embeddings)
semantic_df.columns = [f"emb_{i}" for i in range(semantic_df.shape[1])]

print("Semantic feature shape:", semantic_df.shape)

# Add TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['utterance'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
#(B)DIALOGUE VIEW
import numpy as np

df['is_question'] = df['utterance'].apply(lambda x: 1 if '?' in str(x) else 0)
df['is_first_turn'] = df.groupby('dialogue_id')['turn_id'].transform(lambda x: (x == x.min()).astype(int))
df['is_last_turn']  = df.groupby('dialogue_id')['turn_id'].transform(lambda x: (x == x.max()).astype(int))


#(c) METADATA FEATURE
df['session_id'] = df['dialogue_id']
df['time_of_day'] = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=len(df))

#ALL COMBINED
# Combine everything
combined_df = pd.concat([df, semantic_df, tfidf_df], axis=1)

print("Final combined shape:", combined_df.shape)
print(combined_df.head(2))

# Save for Phase 4 (Clustering)
combined_df.to_csv('/home/samruddhi/Project/data/phase3_features.csv', index=False)
print(" Features saved to phase3_features.csv")
