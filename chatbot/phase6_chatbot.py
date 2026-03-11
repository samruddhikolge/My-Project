import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Centroids
# -----------------------------
CENTROID_PATH = "/home/samruddhi/Project/data/cluster_centroids.npy"
centroids = np.load(CENTROID_PATH, allow_pickle=True).item()

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Dialogue State
# -----------------------------
state = {
    "domain": None,
    "slots": {
        "hotel": {
            "city": None,
            "price": None,
            "amenities": None
        },
        "restaurant": {
            "city": None,
            "cuisine": None,
            "time": None
        },
        "train": {
            "from_city": None,
            "to_city": None,
            "day": None,
            "time": None
        }
    }
}

last_domain = None  # CRITICAL FIX

# -----------------------------
# Domain Detection
# -----------------------------
def detect_domain(text):
    text = text.lower()
    if "hotel" in text:
        return "hotel"
    if "restaurant" in text or "dine" in text:
        return "restaurant"
    if "train" in text:
        return "train"
    return None

# -----------------------------
# Slot Filling Logic
# -----------------------------
def update_slots(text, domain):
    text = text.lower()

    if domain == "hotel":
        if state["slots"]["hotel"]["city"] is None:
            state["slots"]["hotel"]["city"] = text
            return
        if state["slots"]["hotel"]["price"] is None and any(
            p in text for p in ["cheap", "moderate", "expensive"]
        ):
            state["slots"]["hotel"]["price"] = text
            return
        if state["slots"]["hotel"]["amenities"] is None:
            state["slots"]["hotel"]["amenities"] = text
            return

    if domain == "restaurant":
        if state["slots"]["restaurant"]["city"] is None:
            state["slots"]["restaurant"]["city"] = text
            return
        if state["slots"]["restaurant"]["cuisine"] is None:
            state["slots"]["restaurant"]["cuisine"] = text
            return
        if state["slots"]["restaurant"]["time"] is None:
            state["slots"]["restaurant"]["time"] = text
            return

    if domain == "train":
        if state["slots"]["train"]["from_city"] is None:
            state["slots"]["train"]["from_city"] = text
            return
        if state["slots"]["train"]["to_city"] is None:
            state["slots"]["train"]["to_city"] = text
            return
        if state["slots"]["train"]["day"] is None:
            state["slots"]["train"]["day"] = text
            return
        if state["slots"]["train"]["time"] is None:
            state["slots"]["train"]["time"] = text
            return

# -----------------------------
# Next Question Selector
# -----------------------------
def next_question(domain):
    for slot, value in state["slots"][domain].items():
        if value is None:
            if domain == "hotel":
                if slot == "city":
                    return "Which city do you want the hotel in?"
                if slot == "price":
                    return "What price range are you looking for?"
                if slot == "amenities":
                    return "Any amenities like WiFi or parking?"
            if domain == "restaurant":
                if slot == "city":
                    return "Which city would you like to dine in?"
                if slot == "cuisine":
                    return "What type of cuisine would you like?"
                if slot == "time":
                    return "What time would you like the reservation?"
            if domain == "train":
                if slot == "from_city":
                    return "From which city are you traveling?"
                if slot == "to_city":
                    return "Where do you want to go?"
                if slot == "day":
                    return "Which day are you traveling?"
                if slot == "time":
                    return "What time would you like to travel?"
    return None

# -----------------------------
# Intent Matching
# -----------------------------
def match_intent(text):
    emb = model.encode([text])[0]
    scores = {
        cid: cosine_similarity([emb], [vec])[0][0]
        for cid, vec in centroids.items()
    }
    best_cluster = max(scores, key=scores.get)
    return best_cluster, scores[best_cluster]

# -----------------------------
# Chatbot Reply Logic
# -----------------------------
def chatbot_reply(text):
    global state, last_domain

    # 1 Detect domain
    domain = detect_domain(text)
    new_domain_detected = False

    if domain and domain != state["domain"]:
        state["domain"] = domain
        new_domain_detected = True

    # 2 Intent similarity (for reporting only)
    cluster, confidence = match_intent(text)

    # 3 Slot filling (ONLY after domain is set)
    if state["domain"]:
        if not new_domain_detected:  # KEY FIX
            update_slots(text, state["domain"])

        question = next_question(state["domain"])
        if question:
            return cluster, confidence, question

        return (
            cluster,
            confidence,
            f"I have all the details. Processing your {state['domain']} request."
        )

    #  Low confidence only when no task active
    if confidence < 0.25:
        return cluster, confidence, "Could you please clarify your request?"

    return cluster, confidence, "How can I help you?"

# -----------------------------
# Chat Loop
# -----------------------------
print("\n Intent-Aware Chatbot (Phase 6 – FINAL)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    cid, conf, reply = chatbot_reply(user_input)
    print(f"[Intent Cluster: {cid} | Confidence: {conf:.2f}]")
    print("Bot:", reply)
