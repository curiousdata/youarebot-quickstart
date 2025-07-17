from uuid import uuid4
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

from app.models import GetMessageRequestModel

default_echo_bot_url = "http://localhost:6872"
st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("# Echo bot with Classification 🚀")
st.sidebar.markdown("# Echo bot 🚀")

# Initialize session state variables
if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid4())
    st.session_state.predictions = []
    st.session_state.true_labels = []
    st.session_state.metrics = {"accuracy": [], "log_loss": []}

with st.sidebar:

    detection_threshold = st.slider(
        "Detection threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Threshold for classifying a message as bot or human",
    )

    if st.button("Reset"):
        st.session_state.pop("messages", None)
        st.session_state.dialog_id = str(uuid4())
        st.session_state.predictions = []
        st.session_state.true_labels = []
        st.session_state.metrics = {"accuracy": [], "log_loss": []}

    echo_bot_url = st.text_input(
        "Bot url", key="echo_bot_url", value=default_echo_bot_url, disabled=True
    )

    dialog_id = st.text_input("Dialog id", key="dialog_id", disabled=True)
    
    # Add selector for ground truth labeling
    label_options = {"Human": 0, "Bot": 1}
    default_label = "Human"
    selected_label = st.radio("I am a:", options=list(label_options.keys()), 
                             index=list(label_options.keys()).index(default_label))
    st.session_state.current_true_label = label_options[selected_label]
    
    # Display metrics in sidebar
    st.subheader("Session Metrics")
    if st.session_state.predictions:
        avg_accuracy = np.mean(st.session_state.metrics["accuracy"]) if st.session_state.metrics["accuracy"] else 0
        avg_log_loss = np.mean(st.session_state.metrics["log_loss"]) if st.session_state.metrics["log_loss"] else 0
        st.metric("Average Accuracy", f"{avg_accuracy:.2f}")
        st.metric("Average Log Loss", f"{avg_log_loss:.4f}")
        
        # Plot metrics
        if len(st.session_state.metrics["accuracy"]) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            ax1.plot(st.session_state.metrics["accuracy"])
            ax1.set_title("Accuracy")
            ax1.set_ylim(0, 1)
            
            ax2.plot(st.session_state.metrics["log_loss"])
            ax2.set_title("Log Loss")
            fig.tight_layout()
            st.pyplot(fig)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Type something", "probability": None}]

# Display messages with probability
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "probability" in msg and msg["probability"] is not None:
            prob_percentage = msg["probability"] * 100
            prob_color = "red" if prob_percentage > detection_threshold * 100 else "green"
            st.markdown(f"<span style='color:{prob_color}'>Bot probability: {prob_percentage:.2f}%</span>", 
                       unsafe_allow_html=True)

if message := st.chat_input():
    # Create user message with UUID
    message_id = str(uuid4())
    user_msg = {"role": "user", "content": message}
    st.session_state["messages"].append(user_msg)
    
    # Call /predict for user message
    prediction_response = requests.post(
        echo_bot_url + "/predict",
        json={
            "text": message,
            "dialog_id": dialog_id,
            "id": message_id,
            "participant_index": 0
        },
    )
    
    # Process prediction result
    prediction_result = prediction_response.json()
    user_probability = prediction_result["is_bot_probability"]
    
    # Update user message with probability
    st.session_state["messages"][-1]["probability"] = user_probability
    
    # Store prediction and update metrics
    st.session_state.predictions.append(user_probability)
    st.session_state.true_labels.append(st.session_state.current_true_label)
    
    # Calculate metrics
    if len(st.session_state.predictions) > 0:
        # Accuracy (threshold at 0.5)
        predicted_labels = [1 if p > detection_threshold else 0 for p in st.session_state.predictions]
        accuracy = sum(p == t for p, t in zip(predicted_labels, st.session_state.true_labels)) / len(predicted_labels)
        st.session_state.metrics["accuracy"].append(accuracy)
        
        # Log loss
        if len(st.session_state.true_labels) > 1:
            try:
                logloss = log_loss(st.session_state.true_labels, st.session_state.predictions)
                st.session_state.metrics["log_loss"].append(logloss)
            except:
                # Handle case where all labels are the same
                st.session_state.metrics["log_loss"].append(0.0)
    
    # Get bot response (same as original code)
    response = requests.post(
        echo_bot_url + "/get_message",
        json=GetMessageRequestModel(
            dialog_id=dialog_id, last_msg_text=message, last_message_id=message_id
        ).model_dump(),
    )
    
    json_response = response.json()
    bot_message = json_response["new_msg_text"]
    
    # Get prediction for bot response
    bot_message_id = str(uuid4())
    bot_prediction_response = requests.post(
        echo_bot_url + "/predict",
        json={
            "text": bot_message,
            "dialog_id": dialog_id,
            "id": bot_message_id,
            "participant_index": 1
        },
    )
    
    bot_prediction_result = bot_prediction_response.json()
    bot_probability = bot_prediction_result["is_bot_probability"]
    
    # Add bot message with probability
    assistant_msg = {"role": "assistant", "content": bot_message, "probability": bot_probability}
    st.session_state["messages"].append(assistant_msg)
    
    st.rerun()