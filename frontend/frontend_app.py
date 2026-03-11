import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
CENTROID_PATH = "/home/samruddhi/Project/data/cluster_centroids.npy"
CITY_IMAGE_DIR = "/home/samruddhi/Project/frontend_assets/cities"

# City → image path mapping
CITY_IMAGES = {
    "Cambridge": os.path.join(CITY_IMAGE_DIR, "cambridge.jpg"),
    "London": os.path.join(CITY_IMAGE_DIR, "london.jpg"),
    "Manchester": os.path.join(CITY_IMAGE_DIR, "manchester.jpg"),
    "Birmingham": os.path.join(CITY_IMAGE_DIR, "birmingham.jpg"),
}

# -----------------------------
# Load model + centroids
# -----------------------------
centroids = np.load(CENTROID_PATH, allow_pickle=True).item()
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Dialogue State
# -----------------------------
def init_state():
    return {
        "domain": None,
        "last_bot_question": None,   # helps UI-based slot entry
        "slots": {
            "hotel": {"city": None, "price": None, "amenities": None},
            "restaurant": {"city": None, "cuisine": None, "time": None},
            "train": {"from_city": None, "to_city": None, "day": None, "time": None}
        },
        "chat": []
    }

if "bot_state" not in st.session_state:
    st.session_state.bot_state = init_state()

state = st.session_state.bot_state


# -----------------------------
# Domain Detection
# -----------------------------
def detect_domain(text: str):
    t = text.lower()
    if "hotel" in t:
        return "hotel"
    if "restaurant" in t or "dine" in t:
        return "restaurant"
    if "train" in t:
        return "train"
    return None


# -----------------------------
# Intent Matching (cluster + confidence)
# -----------------------------
def match_intent(text: str):
    emb = model.encode([text])[0]
    scores = {
        cid: float(cosine_similarity([emb], [vec])[0][0])
        for cid, vec in centroids.items()
    }
    best = max(scores, key=scores.get)
    return int(best), float(scores[best])


# -----------------------------
# Slot Filling
# -----------------------------
def update_slots(text: str, domain: str):
    text = text.strip()

    if domain == "hotel":
        if state["slots"]["hotel"]["city"] is None:
            state["slots"]["hotel"]["city"] = text
            return
        if state["slots"]["hotel"]["price"] is None:
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
# Next Question
# -----------------------------
def next_question(domain: str):
    slots = state["slots"][domain]
    for slot, val in slots.items():
        if val is None:
            if domain == "hotel":
                return {
                    "city": "Which city do you want the hotel in?",
                    "price": "What price range are you looking for? (cheap/moderate/expensive)",
                    "amenities": "Any amenities like WiFi or parking?"
                }[slot]

            if domain == "restaurant":
                return {
                    "city": "Which city would you like to dine in?",
                    "cuisine": "What type of cuisine would you like?",
                    "time": "What time would you like the reservation?"
                }[slot]

            if domain == "train":
                return {
                    "from_city": "From which city are you traveling?",
                    "to_city": "Where do you want to go?",
                    "day": "Which day are you traveling?",
                    "time": "What time would you like to travel?"
                }[slot]
    return None


# -----------------------------
# Core Chatbot Reply
# -----------------------------
def chatbot_reply(user_text: str):
    # 1) Detect domain
    domain = detect_domain(user_text)
    new_domain = False

    if domain and domain != state["domain"]:
        state["domain"] = domain
        new_domain = True

    # 2) Cluster inference (for UI)
    cluster, conf = match_intent(user_text)

    # 3) Slot filling when domain active
    if state["domain"]:
        # IMPORTANT FIX:
        # do not fill slots on the intent-trigger utterance
        if not new_domain:
            update_slots(user_text, state["domain"])

        q = next_question(state["domain"])
        if q:
            state["last_bot_question"] = q
            return cluster, conf, q

        final_msg = f"I have all the details. Processing your {state['domain']} request."
        state["last_bot_question"] = final_msg
        return cluster, conf, final_msg

    # 4) No domain detected
    if conf < 0.25:
        msg = "Could you clarify your request? (hotel / restaurant / train)"
        state["last_bot_question"] = msg
        return cluster, conf, msg

    msg = "How can I help you today?"
    state["last_bot_question"] = msg
    return cluster, conf, msg


# -----------------------------
# UI-based Slot Selection (City with images)
# -----------------------------
def show_city_selector():
    """
    If city is needed for hotel/restaurant, show a dropdown + image + confirm button.
    For train, show from_city and to_city selectors (2 dropdowns).
    """

    dom = state["domain"]
    if dom is None:
        return

    st.markdown("---")
    st.subheader(" Smart Slot Input (UI)")

    # Helper: show city image preview
    def show_city_preview(city):
        img = CITY_IMAGES.get(city)
        if img and os.path.exists(img):
            st.image(img, caption=city, use_container_width=True)
        else:
            st.info(f"No image found for {city}. Add: {img}")

    cities = list(CITY_IMAGES.keys())

    # HOTEL CITY
    if dom == "hotel" and state["slots"]["hotel"]["city"] is None:
        st.write(" **Hotel City Selection**")
        city = st.selectbox("Select city", cities, key="hotel_city_select")
        show_city_preview(city)

        if st.button("Confirm Hotel City"):
            push_user_text(city)

    # RESTAURANT CITY
    elif dom == "restaurant" and state["slots"]["restaurant"]["city"] is None:
        st.write("**Restaurant City Selection**")
        city = st.selectbox("Select city", cities, key="restaurant_city_select")
        show_city_preview(city)

        if st.button(" Confirm Restaurant City"):
            push_user_text(city)

    # TRAIN FROM + TO
    elif dom == "train":
        if state["slots"]["train"]["from_city"] is None:
            st.write(" **Train Source City**")
            from_city = st.selectbox("From city", cities, key="train_from_city_select")
            show_city_preview(from_city)

            if st.button("Confirm Source City"):
                push_user_text(from_city)

        elif state["slots"]["train"]["to_city"] is None:
            st.write(" **Train Destination City**")
            to_city = st.selectbox("To city", cities, key="train_to_city_select")
            show_city_preview(to_city)

            if st.button("Confirm Destination City"):
                push_user_text(to_city)


def push_user_text(text: str):
    """
    Adds selection to chat as if user typed it,
    runs chatbot logic, stores response.
    """
    state["chat"].append({"role": "user", "text": text})

    cluster, conf, reply = chatbot_reply(text)

    state["chat"].append({
        "role": "bot",
        "text": reply,
        "cluster": cluster,
        "conf": conf
    })

    st.rerun()


# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Intent Discovery Frontend", page_icon="🤖", layout="wide")

st.title(" Intent Decomposition & Clustering Frontend")
st.caption("Unsupervised intent discovery using SBERT embeddings + K-Means (k=6) and slot-filling dialogue intelligence.")

left, right = st.columns([2.2, 1])

# -----------------------------
# Right: Dialogue State Panel
# -----------------------------
with right:
    st.subheader(" Dialogue State")

    st.write("**Active Domain:**", state["domain"])
    if state["domain"]:
        st.write("**Slots:**")
        st.json(state["slots"][state["domain"]])
    else:
        st.info("No active domain yet. Try: 'book a hotel'")

    st.write("**Last Bot Question:**")
    st.code(state["last_bot_question"] if state["last_bot_question"] else "—")

    if st.button("🔄Reset Conversation"):
        st.session_state.bot_state = init_state()
        st.rerun()

    # Show smart city selector in side panel (optional)
    show_city_selector()

# -----------------------------
# Left: Chat Panel
# -----------------------------
with left:
    st.subheader("💬 Chat Interface")

    # Display chat history
    for msg in state["chat"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            st.chat_message("assistant").write(msg["text"])
            st.caption(f"Intent Cluster: {msg['cluster']} | Confidence: {msg['conf']:.2f}")

    # Input box
    user_input = st.chat_input("Type your message...")
    if user_input:
        push_user_text(user_input)
