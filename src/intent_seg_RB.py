import pandas as pd
import re

# Load the preprocessed dialogue data (user turns)
df = pd.read_csv('/home/samruddhi/Project/data/intents.csv')

def rule_based_segment(text):
    """
    Split text into multiple intent phrases using simple linguistic rules.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Common connectors that separate intents
    connectors = [
        r'\band\b', r'\bor\b', r'\bthen\b', r'\balso\b', 
        r'\bafter that\b', r'\bbesides\b', r'\bin addition\b'
    ]

    # Combine them into one regex pattern
    pattern = '|'.join(connectors)

    # Split sentence
    segments = re.split(pattern, text, flags=re.IGNORECASE)

    # Clean and return
    return [seg.strip() for seg in segments if seg.strip()]

# Apply rule-based segmentation
segmented_data = []

for _, row in df.iterrows():
    text = row['user_utterance']
    dialogue_id = row['dialogue_id']
    turn_id = row['turn_id']
    intents = row['intents']

    segments = rule_based_segment(text)
    if len(segments) == 0:
        segments = [text]

    for i, seg in enumerate(segments):
        segmented_data.append({
            'dialogue_id': dialogue_id,
            'turn_id': turn_id,
            'segment_id': i,
            'utterance_segment': seg,
            'intents': intents
        })

seg_df = pd.DataFrame(segmented_data)
output_csv = '/home/samruddhi/Project/data/intent_segments_RB.csv'
seg_df.to_csv(output_csv, index=False)

print(f"Rule-based segmentation complete: {len(seg_df)} segments saved.")
print(f"Saved to {output_csv}")
print(seg_df.head(10))
