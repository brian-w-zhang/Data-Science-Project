from PIL import Image
import streamlit as st
from pathlib import Path

from inference.service import DisasterClassifier

@st.cache_resource
def get_service() -> DisasterClassifier:
    # Adjust paths to where you saved checkpoints in your project
    root = Path(__file__).parent.parent  # project_root/
    text_dir      = root / "checkpoints" / "text_branch"
    vision_path   = root / "checkpoints" / "vision_brain.pth"
    fusion_path   = root / "checkpoints" / "fusion_brain.pth"

    return DisasterClassifier(
        text_model_dir=str(text_dir),
        vision_weights_path=str(vision_path),
        fusion_weights_path=str(fusion_path),
        device="cpu",  # or "cuda" if available and appropriate
    )

service = get_service()

fake_tweet = """BREAKING: A confetti volcano has erupted in New Avalon, showering the city in biodegradable glitter. Authorities advise sunglasses and leaf blowers. No injuries—just sparkle. Stay fabulous, stay safe. #NewAvalon #ConfettiCano"""
real_tweet = """[ALERT] Gas explosion in East Riverton at 7:45 PM. 5 injured, 2 missing (confirmed by Fire Dept). Avoid Harbor Ave and 3rd St. Temporary shelter at Douglas Rec Center, 1120 Pine. Family reunification: 555-0134 or emergency.riverton.gov. Updates every 30 min. #Riverton"""

st.title("Disaster Tweet Classifier")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add the assistant's greeting to chat history on first load
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! Please write a tweet or start with these examples."
    })

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image":
                    st.image(part["data"])
        else:
            st.markdown(message["content"])

        # Attach example buttons to the first assistant message
        if message["role"] == "assistant" and i == 0:
            if st.button("Fake disaster"):
                st.session_state.chat_input = fake_tweet
            if st.button("Realistic disaster"):
                st.session_state.chat_input = real_tweet

# React to user input
if prompt := st.chat_input(
    "Write tweet and/or attach an image",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
    key="chat_input"
):
    user_content = []

    # ChatInputValue always has .text and .files when accept_file=True
    text = prompt.text.strip() if prompt.text else ""
    files = prompt.files if prompt.files else []

    if text:
        user_content.append({"type": "text", "text": text})

    images = []
    for file in files:
        image_bytes = file.read()
        user_content.append({"type": "image", "data": image_bytes})
        images.append(image_bytes)

    # Display user message
    with st.chat_message("user"):
        for part in user_content:
            if part["type"] == "text":
                st.markdown(part["text"])
            elif part["type"] == "image":
                st.image(part["data"])

    st.session_state.messages.append({"role": "user", "content": user_content})

    images = []
    for file in files:
        pil_img = Image.open(file).convert("RGB")
        user_content.append({"type": "image", "data": pil_img})
        images.append(pil_img)

    result = service.predict(
    text=text or None,
    image=images[0] if images else None,
    )
    response_text = f"**{result['label']}** ({result['confidence']:.1%} confidence)"

    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})