import logging
from flask import Flask, request, jsonify
from transformers import pipeline
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import torch

# Logging configuration
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# Telegram Bot Token
TELEGRAM_TOKEN = "7782403497:AAHCFUOHncSZPIGFBbO0JfHgJfhx2cQHtr8"  # Replace with your Telegram Bot Token

# Define Telegram command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! I am your AI assistant. Ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text

    # Construct the prompt
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant."},
        {"role": "user", "content": user_message.strip()},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate response using the model
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        ai_response = outputs[0]["generated_text"]

        # Post-processing: Remove unnecessary tags and extra content
        if "<|assistant|>" in ai_response:
            ai_response = ai_response.split("<|assistant|>")[-1].strip()

        await update.message.reply_text(ai_response)
    except Exception as e:
        await update.message.reply_text(f"Error generating response: {str(e)}")

# Define error handler
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Update {update} caused error {context.error}")

# Start the Telegram bot
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error)

    application.run_polling()

# Initialize the Flask app
app = Flask(__name__)

# Load the TinyLlama model and pipeline
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

try:
    pipe = pipeline(
        task="text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model or pipeline: {str(e)}")

if __name__ == "__main__":
    main()
