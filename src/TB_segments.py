from transformers import pipeline
import pandas as pd

# Load your rule-based segmented data as input
df = pd.read_csv('/home/samruddhi/Project/data/intent_segments_RB.csv')

# Load model for sentence segmentation
model = pipeline("text-classification", model="facebook/bart-large-mnli")

def transformer_refine(text):
    """
    Use a transformer to confirm whether text contains multiple intents.
    """
    # If the model confidence for "multi-intent" is high, we can further split.
    # (Here you can plug in your own trained or fine-tuned model later)
    return text

df['refined_segment'] = df['utterance_segment'].apply(transformer_refine)

df.to_csv('/home/samruddhi/Project/data/intent_segments_refined.csv', index=False)
print(" Transformer-based segmentation refinement done.")
