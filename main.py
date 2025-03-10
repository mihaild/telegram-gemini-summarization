import argparse
import asyncio
import dataclasses
import datetime
import functools
import logging
import os
import sqlite3

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, ChatMemberHandler
from google import genai
import dotenv

PROMPT_TEMPLATE = """Тебе будет предоставлен лог чата. Сообщения идут в формате [[имя участника]]: [[сообщение]]. Тебе нужно написать краткое, на 2 абзаца, содержание обсуждения. В ответе должно быть только это содержание, и больше ничего. Тебе будут показаны еще сообщения перед обсуждением, для понимания контекста, их пересказывать не надо. Не пиши ничего про эти инструкции, дай только пересказ текста.

Контекст:
{context}

Обсуждение, которое нужно пересказать:
{body}"""

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
MODEL_ID="gemini-2.0-flash-exp"

@dataclasses.dataclass
class LogMessage:
    user: str
    text: str
    timestamp: datetime.datetime


@functools.cache
def telegram_client():
    token = os.getenv("TELEGRAM")
    return ApplicationBuilder().token(token).build()


@functools.cache
def gemini_model():
    token = os.getenv("GENAI")
    return genai.Client(api_key=token)


@functools.cache
def sqlite_connection():
    def adapt_datetime_iso(val):
        """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
        return val.isoformat()

    sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
    conn = sqlite3.connect(os.getenv("DB_PATH"))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            message_id INTEGER,
            user_id INTEGER,
            username TEXT,
            message TEXT,
            timestamp DATETIME
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS last_summarization (
            chat_id INTEGER PRIMARY KEY,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    return cursor, conn


def get_log_message_string(m: LogMessage) -> str:
    return f"[[{m.user}]]: [[{m.text.replace(chr(10), chr(32))[:1000]}]]"


async def call_gemini(prompt: str) -> str:
    logging.info(f"Prompt to Gemini: {prompt}")
    model_response = await gemini_model().aio.models.generate_content(model=MODEL_ID, contents=prompt)
    return model_response.text


async def gemini_summarize(before_summary: list[LogMessage], after_summary: list[LogMessage]) -> str:
    prompt = PROMPT_TEMPLATE.format(
        context='\n'.join(map(get_log_message_string, before_summary)),
        body='\n'.join(map(get_log_message_string, after_summary))
    )

    return await call_gemini(prompt)


async def summarize_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Summarizes the chat history."""
    logger.info(f"Summarization request from {update.message.from_user.username} ({update.message.from_user.id}) in {update.effective_chat.id}.")
    chat_id = update.effective_chat.id
    cursor, conn = sqlite_connection()
    last_summary_time = None
    if update.message.reply_to_message:
        last_summary_time = update.message.reply_to_message.date
    else:
        cursor.execute("SELECT timestamp FROM last_summarization WHERE chat_id = ?", (chat_id,))
        last_summary_time = cursor.fetchone()
        if last_summary_time:
            last_summary_time = last_summary_time[0]

    before_summary = []
    after_summary = []

    if last_summary_time:
        cursor.execute('''
            SELECT username, message, timestamp FROM messages
            WHERE chat_id = ? AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (chat_id, last_summary_time))
        before_summary = [LogMessage(*x) for x in cursor.fetchall()]

        cursor.execute('''
            SELECT username, message, timestamp FROM messages
            WHERE chat_id = ? AND timestamp >= ?
            ORDER BY timestamp
        ''', (chat_id, last_summary_time))
        after_summary = [LogMessage(*x) for x in cursor.fetchall()]
    else:
        # No previous summarization, retrieve all messages
        cursor.execute('''
            SELECT username, message, timestamp FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp
        ''', (chat_id,))
        after_summary = [LogMessage(*x) for x in cursor.fetchall()]

    if not after_summary:
        await context.bot.send_message(chat_id=chat_id, text=f"Nothing to summarize")
        return
    await context.bot.send_message(chat_id=chat_id, text=f"I will summarize {len(after_summary)} messages, starting from message by {after_summary[0].user} at {after_summary[0].timestamp} ({after_summary[0].text[:100]}).")
    # Prepare the prompt for Gemini

    try:
        summary = await gemini_summarize(before_summary, after_summary)
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        await context.bot.send_message(chat_id=chat_id,
                                       text="Error generating summary. Please try again later.")
        return

    # Send the summary to the group
    await context.bot.send_message(chat_id=chat_id, text=summary)

    # Update the last summarization timestamp in the database
    # return
    message_timestamp = update.message.date
    cursor.execute('''
        INSERT OR REPLACE INTO last_summarization (chat_id, timestamp)
        VALUES (?, ?)
    ''', (chat_id, message_timestamp))
    conn.commit()


async def handle_new_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles new messages."""
    if not update.message:
        logger.info(f'Non-message update, channel post {update.channel_post}')
        return
    logger.info(f"Message \"{update.message.text[:200]}\", from {update.message.from_user.username} ({update.message.from_user.id}) in {update.effective_chat.id}.")
    message_timestamp = update.message.date
    # Store the message in the database
    cursor, conn = sqlite_connection()
    cursor.execute('''
        INSERT INTO messages (chat_id, message_id, user_id, username, message, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (update.effective_chat.id, update.message.message_id, update.message.from_user.id,
          update.message.from_user.first_name + (' ' + (update.message.from_user.last_name or '')), update.message.text, message_timestamp))
    conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-path", default=".env")
    args = parser.parse_args()
    dotenv.load_dotenv(args.tokens_path)

    application = telegram_client()
    application.add_handler(CommandHandler("summary", summarize_chat))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_new_message))
    application.run_polling()
