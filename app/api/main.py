from fastapi import FastAPI, HTTPException
from transformers import pipeline
import traceback
import time
import random
from functools import lru_cache
from uuid import uuid4
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
import os
import httpx
load_dotenv()

from app.core.logging import app_logger
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction

app = FastAPI()

llm_url = "http://localhost:8080/completion"

# Separate histories for different purposes
chat_history = defaultdict(list)  # For OpenAI chat (dicts with role/content)
classification_history = defaultdict(list)  # For text classification (strings)
participant_classification_history = defaultdict(lambda: defaultdict(list))  # Nested defaultdict for participants

MAX_HISTORY_LENGTH = 10

# Remove this client since we won't be using OpenRouter anymore
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )

@lru_cache(maxsize=1)
def get_zero_shot_pipeline():
    """Load and cache the classification model"""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    """
    This function generates a response using a local LLM
    """
    try:
        app_logger.info(
            f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
        )
        
        # Initialize chat history with system prompt if needed
        if not chat_history[body.dialog_id]:
            chat_history[body.dialog_id] = [
                {
                    "role": "system", 
                    "content": "You are texting like a regular person, not an assistant. Avoid being helpful or formal. Be casual and write like in a normal phone chat."
                }
            ]
        
        # Add user message to chat history
        chat_history[body.dialog_id].append({"role": "user", "content": body.last_msg_text})
        
        # Also add to classification history for future reference
        classification_history[body.dialog_id].append(body.last_msg_text)
        
        # Trim histories if too long
        if len(chat_history[body.dialog_id]) > MAX_HISTORY_LENGTH + 1:  # +1 for system
            system_message = chat_history[body.dialog_id][0]
            messages = chat_history[body.dialog_id][1:]
            chat_history[body.dialog_id] = [system_message] + messages[-(MAX_HISTORY_LENGTH-1):]
            
        if len(classification_history[body.dialog_id]) > MAX_HISTORY_LENGTH:
            classification_history[body.dialog_id].pop(0)
        
        # Prepare messages in a format suitable for your local LLM
        formatted_messages = []
        for msg in chat_history[body.dialog_id]:
            if msg["role"] == "system":
                formatted_messages.append(f"<|system|>\n{msg['content']}")
            elif msg["role"] == "user":
                formatted_messages.append(f"<|user|>\n{msg['content']}")
            elif msg["role"] == "assistant":
                formatted_messages.append(f"<|assistant|>\n{msg['content']}")
        
        # Join all messages
        prompt = "\n".join(formatted_messages)
        prompt += "\n<|assistant|>\n"  # Add the final token to prompt for response
        
        # Call local LLM API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                llm_url,
                json={
                    "prompt": prompt,
                    "temperature": 0.9,
                    "max_tokens": 100,
                },
                timeout=30.0  # Increase timeout for local inference
            )
            
            if response.status_code != 200:
                raise Exception(f"LLM API returned status code {response.status_code}: {response.text}")
            
            response_data = response.json()
            # Adjust this based on your local LLM API response format
            response_text = response_data.get("content", "")  # Change this key if needed
        
        # Add bot response to histories
        chat_history[body.dialog_id].append({"role": "assistant", "content": response_text})
        classification_history[body.dialog_id].append(response_text)

        # realistic response times
        max_total_delay = 2.0
        reading_time = 0.2 + (len(body.last_msg_text) * 0.03)
        thinking_time = random.uniform(1.0, 3.0)
        typing_time = 0.3 + (len(response_text) * 0.05)
        randomness = random.uniform(0.8, 1.2)
        total_delay = (reading_time + thinking_time + typing_time) * randomness
        total_delay = min(total_delay, max_total_delay)
        time.sleep(total_delay)
        
        return GetMessageResponseModel(
            new_msg_text=response_text, dialog_id=body.dialog_id
        )
    except Exception as e:
        app_logger.error(f"Error generating response: {str(e)}\n{traceback.format_exc()}")
        # Fallback to echo bot
        return GetMessageResponseModel(
            new_msg_text=f"Echo: {body.last_msg_text}", dialog_id=body.dialog_id
        )


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Classifies a message as human or bot with participant separation
    """
    try:
        # Always initialize to False at the start of the function
        
        classifier = get_zero_shot_pipeline()

        
        # Add to overall classification history
        classification_history[msg.dialog_id].append(msg.text)
        
        # Add to participant-specific history using participant_index
        participant_key = f"{msg.dialog_id}_{msg.participant_index}"
        participant_classification_history[msg.dialog_id][msg.participant_index].append(msg.text)
        
        # Limit overall history length
        if len(classification_history[msg.dialog_id]) > MAX_HISTORY_LENGTH:
            classification_history[msg.dialog_id].pop(0)
            
        # Limit participant history length
        if len(participant_classification_history[msg.dialog_id][msg.participant_index]) > MAX_HISTORY_LENGTH:
            participant_classification_history[msg.dialog_id][msg.participant_index].pop(0)

        # Get participant-specific text (only messages from this participant)
        participant_text = ";".join(participant_classification_history[msg.dialog_id][msg.participant_index])
        
        # Get dialogue context (alternating messages with participant markers)
        dialogue_pairs = []
        for i in range(0, len(classification_history[msg.dialog_id]), 2):
            if i < len(classification_history[msg.dialog_id]):
                user_msg = f"User: {classification_history[msg.dialog_id][i]}"
                dialogue_pairs.append(user_msg)
            if i+1 < len(classification_history[msg.dialog_id]):
                bot_msg = f"Response: {classification_history[msg.dialog_id][i+1]}"
                dialogue_pairs.append(bot_msg)
        dialogue_context = ". ".join(dialogue_pairs)
        
        # Run classification on BOTH participant text and dialogue context
        participant_result = classifier(
            participant_text, 
            candidate_labels=[
                "written by a human with natural flow, varying sentence lengths, occasional errors, and personal style", 
                "written by an AI with strict punctuation, consistent tone, formulaic responses, precise grammar, and repetitive patterns"
            ],
            hypothesis_template="These messages are {}."
        )
        
        dialogue_result = classifier(
            dialogue_context,
            candidate_labels=[
                "a natural human conversation with realistic back-and-forth exchanges",
                "an AI-generated conversation with formulaic responses"
            ],
            hypothesis_template="This is {}."
        )
        
        # Weight individual style more heavily than dialogue context
        participant_prob = participant_result['scores'][1]
        dialogue_prob = dialogue_result['scores'][1]
        is_bot_probability = (participant_prob * 0.7) + (dialogue_prob * 0.3)
        
        prediction_id = str(uuid4())
        return Prediction(
            id=prediction_id,
            message_id=msg.id,
            dialog_id=msg.dialog_id,
            participant_index=msg.participant_index,
            is_bot_probability=is_bot_probability
        )
    except Exception as e:
        app_logger.error(f"Error in predict: {str(e)}\n{traceback.format_exc()}")
        # Return a fallback prediction
        return Prediction(
            id=str(uuid4()),
            message_id=msg.id,
            dialog_id=msg.dialog_id,
            participant_index=msg.participant_index,
            is_bot_probability=0.5
        )
    
@app.post("/reset_history")
def reset_history(dialog_id: str):
    """Clear all history for a specific dialog"""
    deleted = False
    
    if dialog_id in chat_history:
        del chat_history[dialog_id]
        deleted = True
        
    if dialog_id in classification_history:
        del classification_history[dialog_id]
        deleted = True
        
    if deleted:
        return {"status": "success", "message": f"History cleared for dialog {dialog_id}"}
    return {"status": "not_found", "message": f"No history found for dialog {dialog_id}"}