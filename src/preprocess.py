import os
import json
import pandas as pd

# Define the folder where JSON files are stored
base_path = '/home/samruddhi/Project/data/MultiWOZ_2.2/train'

# Where to save the preprocessed output
output_csv = '/home/samruddhi/Project/data/preprocessed_conversations.csv'

all_turns = []  # collect all turns here

for filename in os.listdir(base_path):
    if filename.endswith('.json'):
        file_path = os.path.join(base_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Each file may contain multiple dialogues
        for dialogue in data:
            dialogue_id = dialogue.get('dialogue_id', filename)

            # Extract turns from each dialogue
            for turn in dialogue.get('turns', []):
                speaker = turn.get('speaker', 'unknown')
                utterance = turn.get('utterance', '').strip()
                turn_id = turn.get('turn_id', -1)

                # Optional: extract active intent if available
                intent = None
                if 'frames' in turn and turn['frames']:
                    for frame in turn['frames']:
                        if 'state' in frame:
                            intent = frame['state'].get('active_intent', None)
                            break  # take the first intent found

                all_turns.append({
                    'dialogue_id': dialogue_id,
                    'turn_id': turn_id,
                    'speaker': speaker,
                    'utterance': utterance,
                    'intent': intent
                })

#create DataFrame only once, after all files processed
df = pd.DataFrame(all_turns)

# Sort for cleaner viewing
df = df.sort_values(by=['dialogue_id', 'turn_id']).reset_index(drop=True)

print(f"Extracted {len(df)} turns.")
print(df.head())

df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
