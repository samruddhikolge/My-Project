
import pandas as pd

# Load the preprocessed data
input_csv = '/home/samruddhi/Project/data/preprocessed_conversations.csv'
output_csv = '/home/samruddhi/Project/data/dialogue_pairs.csv'

df = pd.read_csv(input_csv)

pairs = []

# Go dialogue by dialogue
for dialogue_id, group in df.groupby('dialogue_id'):
    # Sort by turn_id to keep conversation order
    group = group.sort_values('turn_id').reset_index(drop=True)
    
    # Pair user utterances with the following system reply
    for i in range(len(group) - 1):
        current = group.iloc[i]
        next_turn = group.iloc[i + 1]
        
        if current['speaker'].upper() == 'USER' and next_turn['speaker'].upper() == 'SYSTEM':
            pairs.append({
                'dialogue_id': dialogue_id,
                'turn_id': current['turn_id'],
                'user_text': current['utterance'],
                'system_response': next_turn['utterance']
            })

# Convert to DataFrame
pairs_df = pd.DataFrame(pairs)

print(f"Extracted {len(pairs_df)} user–system pairs.")
print(pairs_df.head())

# Save output
pairs_df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
