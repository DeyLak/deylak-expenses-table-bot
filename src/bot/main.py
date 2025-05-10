import logging
import os
import json
import httpx # Import httpx for MCP calls
from typing import List, Any, Tuple, Dict # Import Dict
import re # Import regex for URL validation
import tempfile # For handling temporary audio files
import traceback # For logging errors

from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove # For potential future use
# Import ApplicationBuilder instead of Application
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes 
# --- OpenAI Import --- START
from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError, APIStatusError
# --- OpenAI Import --- END
# --- Faster Whisper Import --- START
from faster_whisper import WhisperModel
# --- Faster Whisper Import --- END

load_dotenv()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Constants ---
USER_SETTINGS_FILE = "user_settings.json"

# Load token from environment variable
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# --- Deepseek Variables --- START
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat") # Default model
# --- Deepseek Variables --- END
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL") # e.g., http://127.0.0.1:8000
# --- Add Model Name Loading ---
# Default to Haiku if not specified
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307") 
# --- End Model Name Loading ---
# Google Apps Script Web App URL (Optional, if using that method)
# GOOGLE_APPS_SCRIPT_URL = os.getenv("GOOGLE_APPS_SCRIPT_URL") 

# --- START Credential Validation ---
missing_vars = []
if not TELEGRAM_BOT_TOKEN:
    missing_vars.append("TELEGRAM_BOT_TOKEN")
# --- Deepseek Key Validation --- START
if not DEEPSEEK_API_KEY:
    missing_vars.append("DEEPSEEK_API_KEY")
# --- Deepseek Key Validation --- END
if not MCP_SERVER_URL:
    missing_vars.append("MCP_SERVER_URL") # Required for core functionality
# No need to strictly validate ANTHROPIC_MODEL_NAME as we have a default
# else: logger.info(f"Using Anthropic model: {ANTHROPIC_MODEL_NAME}")

if missing_vars:
    logger.fatal(f"Required environment variables missing: {', '.join(missing_vars)}")
    exit(1) # Exit if critical configuration is missing
else:
    logger.info("All required environment variables loaded.")
    # Log the model being used
    logger.info(f"Using Deepseek model: {DEEPSEEK_MODEL_NAME}") 
# --- END Credential Validation ---

# Be careful logging sensitive info, only log confirmation if needed
# logger.info(f"ANTHROPIC_API_KEY loaded (checking presence): {bool(ANTHROPIC_API_KEY_LOADED)}")
# logger.info(f"MCP_SERVER_URL: {MCP_SERVER_URL}")

# --- Initialize Deepseek Client (using OpenAI library) --- START
llm_client = None
if DEEPSEEK_API_KEY:
    try:
        llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1" # Point to Deepseek API
        )
        logger.info("Deepseek client initialized successfully (using OpenAI library).")
        # Note: Testing connection like listing models might not work if Deepseek
        # doesn't implement that specific OpenAI endpoint.
    except AuthenticationError as e:
        logger.error(f"Deepseek Authentication Error: Invalid API key? {e}")
    except APIConnectionError as e:
        logger.error(f"Deepseek API Connection Error: Could not connect. {e}")
    except Exception as e:
        logger.error(f"Failed to initialize Deepseek client: {e}")
else:
    logger.warning("DEEPSEEK_API_KEY not found. Deepseek LLM features will be unavailable.")
# --- Initialize Deepseek Client --- END

# --- Load Whisper Model --- START
# Specify the model size (e.g., "tiny", "base", "small", "medium", "large-v2", "large-v3")
# Smaller models are faster and use less memory but are less accurate.
# "base" is a good starting point.
WHISPER_MODEL_SIZE = "base"
# You might need to adjust device ("cuda", "cpu") and compute_type ("float16", "int8")
# based on your hardware. "cpu" and "int8" are generally more compatible.
# Note: faster-whisper might download the model automatically on first run.
logger.info(f"Loading faster-whisper model: {WHISPER_MODEL_SIZE}...")
try:
    # Using compute_type="int8" for potential CPU performance boost
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    logger.info(f"Faster-whisper model '{WHISPER_MODEL_SIZE}' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load faster-whisper model '{WHISPER_MODEL_SIZE}': {e}")
    logger.error("Voice message processing will be unavailable.")
    # Consider adding instructions here on how to download models or install dependencies
    logger.error("Ensure the model files are downloaded or accessible and required dependencies (like ffmpeg for format conversion) are installed.")
    whisper_model = None
# --- Load Whisper Model --- END

# --- User Settings Store --- 
def load_user_settings() -> Dict[str, Dict]:
    """Loads user settings from the JSON file."""
    try:
        with open(USER_SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"'{USER_SETTINGS_FILE}' not found, starting with empty settings.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{USER_SETTINGS_FILE}'. Starting empty.")
        # Consider backing up the broken file here
        return {}

def save_user_settings(settings: Dict[str, Dict]):
    """Saves user settings to the JSON file."""
    try:
        with open(USER_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
    except IOError as e:
        logger.error(f"Failed to save user settings to '{USER_SETTINGS_FILE}': {e}")

# Load initial settings
user_settings = load_user_settings()

# --- User Context Cache (In-memory) ---
# Structure: { settings_key (user_id_str or chat_id_str): { "script_url": url, "participants": [...], "categories": [...] } }
user_context_cache: Dict[str, Dict[str, Any]] = {}

# --- MCP Call Helper --- #
# TODO (Phase 2): Modify this to accept script_url and pass it in arguments
async def call_mcp(function_name: str, arguments: dict) -> dict | None:
    """Calls the specified function on the MCP server."""
    if not MCP_SERVER_URL:
        logger.error("Cannot call MCP: MCP_SERVER_URL is not configured.")
        return {"status": "error", "details": "MCP Server URL not configured"}
    
    mcp_call_url = f"{MCP_SERVER_URL.rstrip('/')}/call"
    # arguments payload will be modified later to include script_url if needed
    request_body = {
        "function_name": function_name,
        "arguments": arguments 
    }
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Calling MCP: {mcp_call_url} with body: {request_body}")
            response = await client.post(mcp_call_url, json=request_body, timeout=15.0)
            response.raise_for_status() # Check for HTTP errors
            
            response_data = response.json()
            logger.info(f"MCP Response ({function_name}): {response_data}")
            
            # Check for application-level errors returned by MCP
            if response_data.get('error'):
                 logger.error(f"MCP function '{function_name}' returned error: {response_data['error']}")
                 # Return the whole error structure from MCP now
                 return {"status": "error", "details": response_data['error']} 
                 
            # Return the successful result part (assuming it's in 'result')
            # If MCP returns { "status": "success", ... }, this needs adjustment
            mcp_result_data = response_data.get('result')
            if mcp_result_data is None:
                 logger.warning(f"MCP response for {function_name} missing 'result' field: {response_data}")
                 # Check if the top-level response indicates success (like from MCP server)
                 if response_data.get("status") == "success":
                     return response_data # Return the whole success dict
                 else:
                     return {"status": "error", "details": "MCP response missing result field."} # Indicate error
            return mcp_result_data # Return just the content of 'result'
            
    except httpx.RequestError as exc:
        logger.error(f"HTTP Request error calling MCP ({function_name}): {exc}")
        return {"status": "error", "details": f"Network error calling MCP: {exc}"} 
    except httpx.HTTPStatusError as exc:
        logger.error(f"HTTP Status error calling MCP ({function_name}): {exc.response.status_code} - {exc.response.text[:200]}")
        return {"status": "error", "details": f"MCP server returned error status {exc.response.status_code}"}
    except json.JSONDecodeError as j_exc:
         logger.error(f"Failed to decode JSON response from MCP ({function_name}): {j_exc}")
         return {"status": "error", "details": "Invalid JSON response from MCP server."}
    except Exception as e:
        logger.exception(f"Unexpected error calling MCP ({function_name}): {e}")
        return {"status": "error", "details": f"Unexpected error calling MCP: {e}"}

# --- Get/Fetch User Context (MODIFIED) --- 
async def get_or_fetch_user_context(user_id: str, chat_id: str, chat_type: str) -> Dict[str, Any]:
    """Gets context (URL, participants, categories) from cache or fetches from MCP.
       Uses user_id for private chats and chat_id for group chats as the settings key.
    """
    global user_context_cache, user_settings
    
    settings_key = user_id if chat_type == "private" else chat_id
    logger.info(f"Determined settings_key: {settings_key} (user: {user_id}, chat: {chat_id}, type: {chat_type})")

    # 1. Get configured URL using settings_key
    user_url = user_settings.get(settings_key, {}).get("script_url")
    if not user_url:
        return {"status": "error", "details": f"Script URL not set for this {'user' if chat_type == 'private' else 'group'}. Key: {settings_key}"}

    # 2. Check cache using settings_key
    cached_context = user_context_cache.get(settings_key)
    if cached_context and cached_context.get("script_url") == user_url:
        logger.info(f"Using cached context for key {settings_key}")
        return {"status": "success", **cached_context}

    # 3. Fetch from MCP if not cached or URL changed
    logger.info(f"Fetching context from MCP for key {settings_key} (URL: {user_url})")
    participants = None
    categories = None
    fetch_error = None
    
    mcp_args = {"script_url": user_url}
    
    participants_result = await call_mcp("getParticipantsForUrl", mcp_args)
    if participants_result and isinstance(participants_result, dict) and participants_result.get('status') == 'success':
        participants = participants_result.get('participants', [])
        logger.info(f"Fetched {len(participants)} participants for key {settings_key}.")
    else:
        error_detail = participants_result.get("details", "Failed to fetch participants") if isinstance(participants_result, dict) else "Unknown error fetching participants"
        logger.warning(f"Failed to fetch participants for key {settings_key}: {error_detail}")
        fetch_error = f"Participants: {error_detail}"
        participants = ["Default Spender"] 
        
    categories_result = await call_mcp("getCategoriesForUrl", mcp_args)
    if categories_result and isinstance(categories_result, dict) and categories_result.get('status') == 'success':
        categories = categories_result.get('categories', [])
        logger.info(f"Fetched {len(categories)} categories for key {settings_key}.")
    else:
        error_detail = categories_result.get("details", "Failed to fetch categories") if isinstance(categories_result, dict) else "Unknown error fetching categories"
        logger.warning(f"Failed to fetch categories for key {settings_key}: {error_detail}")
        if fetch_error: fetch_error += f"; Categories: {error_detail}"
        else: fetch_error = f"Categories: {error_detail}"
        categories = ["Default Category"]
        
    if participants is not None and categories is not None:
         new_context = {
             "script_url": user_url,
             "participants": participants,
             "categories": categories
         }
         user_context_cache[settings_key] = new_context # Use settings_key for cache
         logger.info(f"Updated context cache for key {settings_key}")
         return {"status": "success", **new_context}
    else:
        logger.error(f"Failed to establish context for key {settings_key}. Final fetch error: {fetch_error}")
        return {"status": "error", "details": f"Failed to fetch context from MCP. {fetch_error}"}

# --- Bot Data Initialization (REMOVED - now per-user) ---
# async def initialize_bot_data(context: ContextTypes.DEFAULT_TYPE) -> None: ...

# --- Dynamic Prompt Generation (Uses Provided Context) ---
def generate_system_prompt(participants: list, categories: list) -> str | None:
    """Generates the system prompt using the provided participants and categories lists."""
    # Removed global cache access
    logger.info("Generating system prompt using provided context...")

    # Check if lists are provided and valid
    if not participants or not categories:
        logger.error("Cannot generate prompt: participants or categories lists are missing/empty.")
        return None

    # Format lists for the prompt
    participants_list_str = "\n".join([f"{i}: {name}" for i, name in enumerate(participants)])
    categories_list_str = "\n".join([f"{i}: {name}" for i, name in enumerate(categories)])

    # Construct the updated prompt (content remains the same)
    prompt = f"""
    You are an advanced expense tracking assistant integrated into a Telegram bot. Your goal is to analyze user messages to identify and extract expense details, including complex splitting information.
    When an expense is mentioned, you MUST extract the following fields: 'amount' (float, total expense amount), 'reason' (string, the description), 
    'spenderIndex' (integer, index of the person who paid, usually 0 if not specified otherwise), 
    'categoryIndex' (integer, index based on the category list below), and 'coefficients' (list of numbers or empty strings representing each participant's share or contribution ratio).
    
    You MUST respond ONLY with a valid JSON object containing these extracted fields.
    Example Format: {{"amount": 100.0, "reason": "Coffee split", "spenderIndex": 0, "categoryIndex": 2, "coefficients": [...]}}
    If the user message does not appear to be logging an expense, respond with exactly: {{"is_expense": false}}
    
    **Coefficient Rules (Very Important!):**
    - The 'coefficients' list **MUST ALWAYS** have the exact same number of elements as the 'Available Participants' list provided below. **THIS IS MANDATORY.**
    - The order of elements in 'coefficients' MUST correspond directly to the order of names in 'Available Participants'.
    - Determine the *ratio* of the split between the participants involved based on user instruction.
    - Represent this ratio using the **smallest possible positive integers**.
    - For participants who are *not* involved in a specific split (e.g., didn't contribute a specific amount, or aren't part of a ratio split), use an **EMPTY STRING** ('') in their position.
    - **Special Case: Expense "For" Specific People:** If the user says the expense was "for" certain participants (e.g., "gifts for A and B" or "куличи для Коли и Ксюши"), this implies the cost should be split *only* among those beneficiaries. Assign `1` to each beneficiary's coefficient. Everyone else (all non-beneficiaries, **including the spender unless they are also explicitly named as a beneficiary**) MUST have `""` as their coefficient. Pay close attention to identifying only the beneficiaries mentioned after words like "for" or "для".
    - **Examples (assuming 5 participants A, B, C, D, E):**
        - **Even split among ALL 5 participants:** `[1, 1, 1, 1, 1]`
        - **Even split between 2 participants (A, B) out of 5:** `[1, 1, "", "", ""]`
        - **Split between 2 participants (A, B) in a 2:1 ratio out of 5:** `[2, 1, "", "", ""]`
        - **Specific amounts given (A paid 30, D paid 70) out of 5:** `[30, "", "", 70, ""]`
        - **Expense *for* only B and C (out of 5), paid by A:** `["", 1, 1, "", ""]` 
        - **If NO splitting information is given for 5 participants:** Assume even split among all: `[1, 1, 1, 1, 1]`.
    - Do NOT use fractions or percentages; use the smallest positive integer ratios or the specific amounts if provided.
    
    Use the provided lists to determine the correct index for spenderIndex and categoryIndex.
    Assume spenderIndex 0 ({participants[0] if participants else 'Default Spender'}) if the payer isn't explicitly mentioned.
    
    Available Participants:
    {participants_list_str}
    
    Available Categories:
    {categories_list_str}
    Assign the most appropriate categoryIndex based on the user's description.
    """
    logger.info("Generated system prompt with smallest integer ratio coefficient instructions using provided context.")
    return prompt.strip()

# --- System Prompt for Correcting Last Expense Coefficient --- START
def generate_correction_system_prompt(participants: list) -> str | None:
    """Generates the system prompt for extracting participant and new coefficient for correction."""
    logger.info("Generating correction system prompt using provided participants...")

    if not participants:
        logger.error("Cannot generate correction prompt: participants list is missing/empty.")
        return None

    participants_list_str = "\n".join([f"{i}: {name}" for i, name in enumerate(participants)])

    prompt = f"""
    You are an assistant helping to correct the last recorded expense. The user wants to change a coefficient for a specific participant.
    Analyze the user's message to identify the participant (either by their name or their index from the list below) and the new coefficient value they want to set.
    
    You MUST respond ONLY with a valid JSON object containing these two fields: 
    'participantIdentifier' (string, the name or index of the participant as provided by the user) and 
    'newCoefficientValue' (string or number, the new coefficient value).
    
    Example Format: {{"participantIdentifier": "Коля", "newCoefficientValue": "2"}}
    Another Example: {{"participantIdentifier": "0", "newCoefficientValue": 1.5}}
    If the new coefficient is meant to be empty (not involved), the user might say "пусто", "убрать", or similar. In such cases, 'newCoefficientValue' should be an empty string "".

    Available Participants:
    {participants_list_str}
    
    Extract the participant identifier and the new coefficient value from the user's text following the command.
    User's text will be something like: "Коля - 2" or "0 1.5" or "Саша убрать".
    """
    logger.info("Generated correction system prompt.")
    return prompt.strip()
# --- System Prompt for Correcting Last Expense Coefficient --- END

# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    user_id = str(update.effective_user.id) # Use string for JSON keys
    user_settings.setdefault(user_id, {}) # Ensure user entry exists
    
    await update.message.reply_text(
        "Привет! Я бот для учета расходов. \n"
        "Чтобы начать, укажите URL вашего Google Apps Script с помощью команды: \n"
        "`/set_script_url ВАШ_URL_ВЕБ_ПРИЛОЖЕНИЯ_APPS_SCRIPT`\n\n"
        "После этого вы сможете добавлять расходы, просто написав мне! Используйте `/help` для примеров."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    await update.message.reply_text(
        "Чтобы записать расход, просто опишите его в свободной форме, например:\n"
        " - `Потратил 150 рублей на кофе`\n"
        " - `Добавь продукты 5000 поровну`\n"
        " - `Такси 300 рублей заплатил Саша, раздели 2:1 с Колей`\n"
        " - `Аренда 30000, Коля внес 20000, Саша 10000`\n\n"
        "Используйте `/set_script_url ВАШ_URL` чтобы установить или обновить подключение к Google Таблице."
    )

# Basic URL validation regex (HTTPS required)
URL_REGEX = re.compile(r"^https://script\.google\.com/macros/s/.+/exec$")

async def set_script_url_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sets the Google Apps Script URL for the user or group."""
    user_id = str(update.effective_user.id)
    chat_id = str(update.message.chat.id)
    chat_type = update.message.chat.type

    settings_key = user_id if chat_type == "private" else chat_id
    key_type_str = "user" if chat_type == "private" else "group chat"
    logger.info(f"/set_script_url called by user {user_id} in {key_type_str} {settings_key}.")
    
    if not context.args:
        await update.message.reply_text(f"Пожалуйста, укажите URL после команды. Пример: `/set_script_url ВАШ_URL`")
        return
        
    script_url = context.args[0]
    
    if not URL_REGEX.match(script_url):
        await update.message.reply_text("Неверный формат URL. Укажите действительный URL веб-приложения Google Apps Script (начинается с https://script.google.com/macros/s/ и заканчивается на /exec).")
        return
        
    user_settings.setdefault(settings_key, {}) # Ensure entry exists for the key
    user_settings[settings_key]["script_url"] = script_url
    save_user_settings(user_settings)
    
    logger.info(f"Set script URL for {key_type_str} (key: {settings_key}).")
    
    if settings_key in user_context_cache: # Clear cache for the specific key
        del user_context_cache[settings_key]
        logger.info(f"Cleared context cache for key {settings_key} due to URL update.")
        
    await update.message.reply_text(f"✅ URL Google Apps Script успешно установлен! Теперь я буду использовать его для записи ваших расходов.")

# --- New Command: Remove Last Expense --- START
async def remove_last_expense_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /remove_last_expense command."""
    user_id = str(update.effective_user.id)
    chat_id = str(update.message.chat.id)
    chat_type = update.message.chat.type
    settings_key = user_id if chat_type == "private" else chat_id
    key_type_str = "user" if chat_type == "private" else "group chat"
    
    logger.info(f"/remove_last_expense called by user {user_id} in {key_type_str} {settings_key}.")

    # 1. Get context to find the script URL
    context_result = await get_or_fetch_user_context(user_id, chat_id, chat_type)
    if context_result.get("status") == "error":
        error_detail = context_result.get("details", "Unknown error fetching context.")
        await update.message.reply_text(f"❌ Ошибка при получении настроек для удаления: {error_detail}")
        return

    current_script_url = context_result.get("script_url")
    if not current_script_url:
         logger.error(f"Внутренняя ошибка: не удалось получить URL скрипта для удаления для ключа {settings_key}")
         await update.message.reply_text("Внутренняя ошибка: не удалось получить настройку URL скрипта для выполнения команды.")
         return

    # 2. Call MCP
    mcp_arguments = {"script_url": current_script_url}
    mcp_result = await call_mcp("removeLastExpense", mcp_arguments)

    # 3. Handle MCP Response
    if mcp_result and isinstance(mcp_result, dict) and mcp_result.get("status") == "success":
        # Send a fixed, short success message regardless of details
        logger.info(f"Successfully removed last expense for key {settings_key}.")
        await update.message.reply_text("✅ Последний расход удален.")
    else:
        error_details = "Unknown error." 
        if isinstance(mcp_result, dict):
            error_details = mcp_result.get("details", "Не удалось удалить последний расход через сервер.")
        logger.error(f"MCP call to removeLastExpense failed for key {settings_key}. Result: {error_details}")
        await update.message.reply_text(f"❌ Не удалось удалить последний расход. Ответ сервера: {error_details}")
# --- New Command: Remove Last Expense --- END

# --- New Command: Correct Last Expense Coefficient --- START
async def correct_last_expense_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /correct_last_expense command."""
    user_id = str(update.effective_user.id)
    chat_id = str(update.message.chat.id)
    chat_type = update.message.chat.type
    settings_key = user_id if chat_type == "private" else chat_id
    key_type_str = "user" if chat_type == "private" else "group chat"
    
    logger.info(f"/correct_last_expense called by user {user_id} in {key_type_str} {settings_key}.")

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите участника и новое значение коэффициента после команды. \nПример: `/correct_last_expense Коля - 2` или `/correct_last_expense 0 1.5`")
        return

    user_input_for_correction = " ".join(context.args)

    # 1. Get context (participants, script_url)
    context_result = await get_or_fetch_user_context(user_id, chat_id, chat_type)
    if context_result.get("status") == "error":
        error_detail = context_result.get("details", "Unknown error fetching context.")
        await update.message.reply_text(f"❌ Ошибка при получении настроек для коррекции: {error_detail}")
        return

    participants = context_result.get("participants", [])
    current_script_url = context_result.get("script_url")

    if not participants:
        await update.message.reply_text("Не удалось получить список участников для этой команды. Попробуйте позже.")
        return
    if not current_script_url:
         logger.error(f"Внутренняя ошибка: не удалось получить URL скрипта для коррекции для ключа {settings_key}")
         await update.message.reply_text("Внутренняя ошибка: не удалось получить настройку URL скрипта.")
         return
    if not llm_client:
        await update.message.reply_text("Извините, AI агент сейчас недоступен для обработки команды.")
        return

    # 2. Generate LLM prompt for correction
    correction_system_prompt = generate_correction_system_prompt(participants)
    if not correction_system_prompt:
         await update.message.reply_text("Извините, не удалось подготовить контекст для AI ассистента (ошибка генерации промпта коррекции).")
         return

    # 3. Call LLM to extract parameters
    try:
        logger.info(f"Sending message to LLM ({DEEPSEEK_MODEL_NAME}) for correction parameter extraction (user: {user_id}, chat: {chat_id})...")
        completion = llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            max_tokens=150,
            messages=[
                {"role": "system", "content": correction_system_prompt},
                {"role": "user", "content": user_input_for_correction}
            ]
        )
        ai_response = ""
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            ai_response = completion.choices[0].message.content
        else:
            logger.warning(f"Received unexpected or empty response structure from LLM for correction: {completion}")
            await update.message.reply_text("Не удалось получить ответ от AI для извлечения параметров.")
            return
        logger.info(f"Raw LLM response for correction (user {user_id}, chat {chat_id}): {ai_response}")

        # --- Extract JSON --- 
        json_string_to_parse = ai_response
        if ai_response.strip().startswith("```") and ai_response.strip().endswith("```"):
            json_start = ai_response.find('{'); json_end = ai_response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string_to_parse = ai_response[json_start:json_end+1]
        # --- End Extract JSON ---

        parsed_params = json.loads(json_string_to_parse)
        logger.info(f"Parsed correction params from LLM (user: {user_id}, chat: {chat_id}): {parsed_params}")

        participant_identifier = parsed_params.get("participantIdentifier")
        new_coefficient_value = parsed_params.get("newCoefficientValue")

        if participant_identifier is None or new_coefficient_value is None: # Allow empty string for newCoefficientValue
            await update.message.reply_text(f"AI не смог извлечь все необходимые параметры (участник/значение). Ответ AI: {json_string_to_parse}")
            return

        # 4. Resolve participantIndex
        coefficient_index = -1
        try:
            # Try to parse as an index first
            potential_index = int(str(participant_identifier).strip())
            if 0 <= potential_index < len(participants):
                coefficient_index = potential_index
            else:
                await update.message.reply_text(f"Указанный индекс участника ({potential_index}) вне допустимого диапазона (0-{len(participants)-1}).")
                return
        except ValueError:
            # If not an int, try to find by name (case-insensitive)
            for i, p_name in enumerate(participants):
                if str(participant_identifier).strip().lower() == p_name.lower():
                    coefficient_index = i
                    break
            if coefficient_index == -1:
                await update.message.reply_text(f"Участник '{participant_identifier}' не найден. Доступные участники: {', '.join(participants)}.")
                return
        
        # 5. Validate newCoefficientValue (allow numbers or empty string)
        if not (isinstance(new_coefficient_value, (int, float)) or new_coefficient_value == ""):
            try:
                new_coefficient_value = float(new_coefficient_value) # Try to convert if it's a numeric string
            except (ValueError, TypeError):
                if str(new_coefficient_value).strip() != "": # If it's not empty string after trying to convert
                    await update.message.reply_text(f"Неверное значение для нового коэффициента: '{new_coefficient_value}'. Допускаются числа или пустое значение для удаления.")
                    return
        
        # 6. Call MCP
        mcp_arguments = {
            "script_url": current_script_url,
            "coefficientIndex": coefficient_index,
            "coefficientValue": new_coefficient_value
        }
        mcp_result = await call_mcp("correctLastExpenseCoefficient", mcp_arguments)

        # 7. Handle MCP Response
        if mcp_result and isinstance(mcp_result, dict) and mcp_result.get("status") == "success":
            corrected_participant_name = participants[coefficient_index]
            logger.info(f"Successfully corrected last expense coefficient for participant {corrected_participant_name} (index {coefficient_index}) to {new_coefficient_value} for key {settings_key}.")
            
            display_coeff_value = new_coefficient_value if new_coefficient_value != "" else "(пусто)"
            success_message = f"✅ Коэффициент для участника '{corrected_participant_name}' в последнем расходе успешно изменен на '{display_coeff_value}'."
            await update.message.reply_text(success_message)
        else:
            error_details = "Unknown error." 
            if isinstance(mcp_result, dict):
                error_details = mcp_result.get("details", "Не удалось изменить коэффициент через сервер.")
            logger.error(f"MCP call to correctLastExpenseCoefficient failed for key {settings_key}. Result: {error_details}")
            await update.message.reply_text(f"❌ Не удалось изменить коэффициент. Ответ сервера: {error_details}")

    except json.JSONDecodeError:
        logger.error(f"Не удалось декодировать JSON ответ от AI для коррекции (user {user_id}/chat {chat_id}): {json_string_to_parse}")
        await update.message.reply_text(f"Хм, не смог обработать параметры для коррекции. Ответ AI: {json_string_to_parse}")
    except APIConnectionError as e:
        logger.error(f"LLM API connection error during correction for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Извините, не удалось подключиться к AI агенту для коррекции.")
    except RateLimitError as e:
        logger.error(f"LLM rate limit exceeded during correction for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Извините, AI агент сейчас занят (коррекция). Пожалуйста, попробуйте позже.")
    except AuthenticationError as e: # Should be caught by initial client check, but good to have
        logger.error(f"LLM authentication error during correction for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Ошибка аутентификации с AI агентом (коррекция).")
    except APIStatusError as e:
        logger.error(f"LLM API status error during correction for user {user_id}/chat {chat_id}: {e.status_code} - {e.response}")
        await update.message.reply_text("Извините, возникла проблема с AI агентом (коррекция).")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during correction command for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Извините, произошла непредвиденная ошибка при выполнении команды коррекции.")
# --- New Command: Correct Last Expense Coefficient --- END

# --- Refactored Expense Processing Logic (MODIFIED) --- 
async def process_expense_text(text_to_process: str, user_id: str, chat_id: str, chat_type: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes the transcribed text to extract and log expense details."""
    # user_id is still useful for logging who initiated in a group
    logger.info(f"Processing text for user {user_id} in chat {chat_id} (type: {chat_type}): '{text_to_process[:100]}...'")

    if not llm_client:
        await update.message.reply_text("Извините, AI агент сейчас недоступен.")
        return

    # Pass user_id, chat_id, and chat_type to context fetching
    user_context_result = await get_or_fetch_user_context(user_id, chat_id, chat_type)
    
    if user_context_result.get("status") == "error":
        error_detail = user_context_result.get("details", "Unknown error fetching context.")
        await update.message.reply_text(f"❌ Ошибка при получении настроек: {error_detail}")
        return
        
    participants = user_context_result.get("participants", [])
    categories = user_context_result.get("categories", [])
    current_user_url = user_context_result.get("script_url")
    
    if not current_user_url:
         settings_key_for_error = user_id if chat_type == "private" else chat_id
         logger.error(f"Внутренняя ошибка: не удалось получить настройку URL скрипта для ключа {settings_key_for_error}")
         await update.message.reply_text("Внутренняя ошибка: не удалось получить настройку URL скрипта.")
         return
         
    system_prompt = generate_system_prompt(participants, categories)
    if not system_prompt:
         await update.message.reply_text("Извините, не удалось подготовить контекст для AI ассистента (ошибка генерации промпта).")
         return

    try:
        logger.info(f"Sending message to LLM ({DEEPSEEK_MODEL_NAME}) for expense parsing (user: {user_id}, chat: {chat_id})...")
        completion = llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_process}
            ]
        )
        ai_response = ""
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            ai_response = completion.choices[0].message.content
        else:
            logger.warning(f"Received unexpected or empty response structure from LLM: {completion}")
            ai_response = "(No response text received)"
        logger.info(f"Raw LLM response for user {user_id}, chat {chat_id}: {ai_response}")

        # --- Extract JSON from potentially formatted LLM response --- START
        json_string_to_parse = ai_response # Default to the raw response
        if ai_response.strip().startswith("```") and ai_response.strip().endswith("```"):
            # Find the start and end of the actual JSON content
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string_to_parse = ai_response[json_start:json_end+1]
                logger.info(f"Extracted JSON string from initial response: {json_string_to_parse}")
            else:
                # Could not extract JSON even though it had backticks
                logger.warning(f"Could not extract JSON content from fenced response: {ai_response}")
                # Proceed with the raw response, likely causing a JSONDecodeError below
        # --- Extract JSON from potentially formatted LLM response --- END

        try:
            parsed_data = json.loads(json_string_to_parse)
            logger.info(f"Parsed JSON data (user: {user_id}, chat: {chat_id}): {parsed_data}")
            if isinstance(parsed_data, dict) and parsed_data.get("is_expense") is False:
                 await update.message.reply_text("Не получилось распознать расход. Попробуйте еще раз.") # Changed from "Okay, let me know..."
            elif isinstance(parsed_data, dict) and all(k in parsed_data for k in ["amount", "reason", "spenderIndex", "categoryIndex", "coefficients"]):
                  is_valid = True; error_messages = []
                  coeffs = parsed_data.get("coefficients")
                  num_participants = len(participants); num_categories = len(categories)
                  llm_spender_index = parsed_data.get("spenderIndex"); llm_category_index = parsed_data.get("categoryIndex")

                  if not isinstance(coeffs, list):
                      is_valid = False; error_messages.append("AI не предоставил коэффициенты в виде списка.")
                  elif len(coeffs) != num_participants: 
                      logger.warning(f"LLM Coefficient length mismatch for user {user_id}/chat {chat_id}. Expected {num_participants}, got {len(coeffs)}. Attempting correction...")
                      participants_list_str_for_retry = "\n".join([f"{i}: {name}" for i, name in enumerate(participants)])
                      correction_prompt = f"Your previous JSON response had an incorrect number of coefficients. The 'coefficients' list MUST have exactly {num_participants} elements, matching the participants list below. Please re-analyze the original user message and provide the corrected JSON with the right number of coefficients. Original user message: \n\n`{text_to_process}`\n\nAvailable Participants:\n{participants_list_str_for_retry}" 
                      try:
                          correction_completion = llm_client.chat.completions.create(
                              model=DEEPSEEK_MODEL_NAME, 
                              max_tokens=300, 
                              messages=[{"role": "user", "content": correction_prompt}]
                          )
                          corrected_ai_response = ""
                          if correction_completion.choices and correction_completion.choices[0].message and correction_completion.choices[0].message.content:
                              corrected_ai_response = correction_completion.choices[0].message.content
                          logger.info(f"LLM Correction Response for user {user_id}/chat {chat_id}: {corrected_ai_response}")
                          json_start = corrected_ai_response.find('{'); json_end = corrected_ai_response.rfind('}')
                          if json_start != -1 and json_end != -1 and json_end > json_start:
                              json_string_to_parse = corrected_ai_response[json_start:json_end+1]
                              try:
                                  corrected_parsed_data = json.loads(json_string_to_parse)
                                  corrected_coeffs = corrected_parsed_data.get("coefficients")
                                  if isinstance(corrected_coeffs, list) and len(corrected_coeffs) == num_participants:
                                      logger.info(f"LLM successfully corrected coefficient length for user {user_id}/chat {chat_id}.")
                                      parsed_data = corrected_parsed_data; coeffs = corrected_coeffs
                                  else:
                                      is_valid = False; error_messages.append(f"AI не смог исправить длину списка коэффициентов. Ожидалось {num_participants}, получено {len(corrected_coeffs if isinstance(corrected_coeffs, list) else 'N/A')}.")
                              except json.JSONDecodeError: is_valid = False; error_messages.append("Попытка исправления AI не удалась (Неверный JSON).")
                          else: is_valid = False; error_messages.append("AI correction response did not contain valid JSON format.")
                      except RateLimitError as e_corr:
                          is_valid = False; error_messages.append("Ошибка во время попытки исправления AI (Лимит запросов).")
                          logger.error(f"LLM rate limit during correction: {e_corr}")
                      except APIStatusError as e_corr:
                          is_valid = False; error_messages.append("Ошибка во время попытки исправления AI (Статус API).")
                          logger.error(f"LLM API status error during correction: {e_corr}")
                      except Exception as e_corr:
                          is_valid = False; error_messages.append("Ошибка во время попытки исправления AI.")
                          logger.exception(f"Unexpected OpenAI error during correction: {e_corr}")
 
                  if is_valid:
                      if not isinstance(llm_spender_index, int) or not (0 <= llm_spender_index < num_participants):
                           is_valid = False; error_messages.append(f"Неверный индекс плательщика ({llm_spender_index}). Допустимый диапазон: 0-{num_participants-1}.")
                      if not isinstance(llm_category_index, int) or not (0 <= llm_category_index < num_categories):
                           logger.warning(f"LLM provided invalid category index {llm_category_index} for user {user_id}/chat {chat_id}. Defaulting to 0.")
                           parsed_data["categoryIndex"] = 0

                  if is_valid:
                      mcp_arguments = parsed_data.copy(); mcp_arguments["script_url"] = current_user_url
                      mcp_result = await call_mcp("addExpense", mcp_arguments)
                      if mcp_result and isinstance(mcp_result, dict) and mcp_result.get("status") == "success":
                          try:
                              amount = parsed_data.get("amount"); reason = parsed_data.get("reason")
                              spender_idx = parsed_data.get("spenderIndex"); category_idx = parsed_data.get("categoryIndex")
                              coeffs_data = parsed_data.get("coefficients") # Renamed to avoid conflict with outer coeffs
                              spender_name = participants[spender_idx] if 0 <= spender_idx < num_participants else f"Unknown Index {spender_idx}"
                              category_name = categories[category_idx] if 0 <= category_idx < num_categories else f"Unknown Index {category_idx}"
                              split_details_parts = []

                              if isinstance(coeffs_data, list) and len(coeffs_data) == num_participants:
                                  max_name_len = max(len(name) for name in participants) if participants else 10
                                  
                                  # Check for the simple "split equally among all" case first
                                  is_simple_equal_split = True
                                  for c_val_check in coeffs_data:
                                      if c_val_check != 1 and c_val_check != "1": # LLM might return string "1"
                                          is_simple_equal_split = False
                                          break

                                  if is_simple_equal_split:
                                      split_str = "Разделено поровну между всеми"
                                  else:
                                      # Detailed split with amounts
                                      numeric_coeffs_values = []
                                      for c_val_extract in coeffs_data:
                                          try:
                                              numeric_coeffs_values.append(float(c_val_extract))
                                          except (ValueError, TypeError):
                                              # If it's an empty string or non-numeric, it doesn't contribute to sum of ratios
                                              pass 
                                      
                                      total_coeff_sum = sum(nc for nc in numeric_coeffs_values if isinstance(nc, (int, float)) and nc > 0) # Sum only positive coefficients for ratio

                                      for i, c_val_loop in enumerate(coeffs_data):
                                          part_name = f"{participants[i]}:" if i < num_participants else f"Part_{i}"
                                          padded_name = part_name.ljust(max_name_len + 2) # Adjusted padding slightly
                                          
                                          coeff_display_str = ""
                                          if c_val_loop == "": 
                                              coeff_display_str = "—"
                                          else:
                                              try: 
                                                  num_c_loop = float(c_val_loop)
                                                  # Format coefficient (e.g., 2.5, 2, not 2.0)
                                                  formatted_c_loop = int(num_c_loop) if num_c_loop.is_integer() else f"{num_c_loop:.2f}".rstrip('0').rstrip('.')
                                                  coeff_display_str = str(formatted_c_loop)

                                                  if total_coeff_sum > 0 and amount > 0 and num_c_loop > 0:
                                                      individual_amount = (num_c_loop / total_coeff_sum) * amount
                                                      # Format amount (e.g., 500, 250.75)
                                                      formatted_individual_amount = int(individual_amount) if individual_amount.is_integer() else f"{individual_amount:.2f}"
                                                      coeff_display_str += f" ({formatted_individual_amount})"
                                              except (ValueError, TypeError): 
                                                  coeff_display_str = str(c_val_loop) # Fallback for non-numeric as text
                                          
                                          split_details_parts.append(f"{padded_name}{coeff_display_str}")
                                      joined_split_details = '\n'.join(split_details_parts)
                                      split_str = f"<b>Раздел</b>:\n<code>{joined_split_details}</code>"
                              else:
                                  split_str = f"<b>Коэффициенты</b>: {coeffs_data} (Ошибка длины?)"
                              success_message = (f"✅ Расход записан!\n" + f"---------------------\n" + 
                                               f"<b>Сумма</b>: {amount}\n" + f"<b>Причина</b>: {reason}\n" + 
                                               f"<b>Заплатил(а)</b>: {spender_name}\n" + f"<b>Категория</b>: {category_name}\n\n" + f"{split_str}")
                              await update.message.reply_text(success_message, parse_mode="HTML")
                          except Exception as fmt_e:
                              logger.error(f"Error formatting success message for user {user_id}/chat {chat_id}: {fmt_e}")
                              await update.message.reply_text(f"✅ Расход записан! (Не удалось отформатировать детали. Ответ сервера: {mcp_result})")
                      else:
                          error_details = mcp_result.get("details", "Не удалось записать расход через сервер.") if isinstance(mcp_result, dict) else "Unknown MCP error."
                          logger.error(f"MCP call to addExpense failed for user {user_id}/chat {chat_id}. Result: {error_details}")
                          await update.message.reply_text(f"❌ Не удалось записать расход. Ответ сервера: {error_details}")
                  else:
                      validation_error_text = " \n".join(error_messages)
                      await update.message.reply_text(f"Извините, я понял расход, но возникла проблема с деталями: {validation_error_text}. Пожалуйста, попробуйте еще раз.")
            else:
                 logger.warning(f"Ответ AI для пользователя {user_id}/chat {chat_id} был JSON, но не в ожидаемом формате: {parsed_data}")
                 await update.message.reply_text("Извините, не смог разобрать детали расхода. Можете перефразировать?")
        except json.JSONDecodeError:
            logger.error(f"Не удалось декодировать JSON ответ от AI для пользователя {user_id}/chat {chat_id}: {json_string_to_parse}")
            await update.message.reply_text(f"Хм, не смог обработать это как расход. Ответ AI: {json_string_to_parse}")
    except APIConnectionError as e:
        logger.error(f"LLM API connection error for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Извините, не удалось подключиться к AI агенту.")
    except RateLimitError as e:
        logger.error(f"LLM rate limit exceeded for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Извините, AI агент сейчас занят. Пожалуйста, попробуйте позже.")
    except AuthenticationError as e:
        logger.error(f"LLM authentication error (invalid API key?) for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Ошибка аутентификации с AI агентом. Проверьте API ключ.")
    except APIStatusError as e:
        logger.error(f"LLM API status error for user {user_id}/chat {chat_id}: {e.status_code} - {e.response}")
        await update.message.reply_text("Извините, возникла проблема с AI агентом.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during text processing for user {user_id}/chat {chat_id}: {e}")
        await update.message.reply_text("Извините, произошла непредвиденная ошибка при обработке текста.")
# --- Refactored Expense Processing Logic --- END

# --- Original Text Message Handler --- START
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles non-command text messages by passing them to the expense processor."""
    user_text = update.message.text
    user_id = str(update.effective_user.id)
    chat_id = str(update.message.chat.id)
    chat_type = update.message.chat.type

    logger.info(f"Received text message from user {user_id} in chat {chat_id} (type: {chat_type})")

    if chat_type == "private":
        await process_expense_text(user_text, user_id, chat_id, chat_type, update, context)
    elif chat_type in ["group", "supergroup"]:
        if 'bot_username' not in context.bot_data:
            bot_info = await context.bot.get_me()
            context.bot_data['bot_username'] = f"@{bot_info.username}"
            logger.info(f"Fetched and cached bot username: {context.bot_data['bot_username']}")
        
        bot_username = context.bot_data['bot_username']
        if user_text.startswith(bot_username):
            logger.info(f"Bot was mentioned in group chat {chat_id} by user {user_id}.")
            text_to_process = user_text[len(bot_username):].lstrip()
            if not text_to_process:
                logger.info("Mention was empty, asking for details.")
                await update.message.reply_text("Привет! Чем могу помочь с расходами? Пожалуйста, опишите ваш расход после упоминания меня.")
                return
            await process_expense_text(text_to_process, user_id, chat_id, chat_type, update, context)
        else:
            logger.info(f"Message in group chat {chat_id} did not mention the bot. Ignoring.")
    else:
        logger.info(f"Received message from unhandled chat type: {chat_type} in chat {chat_id}. Ignoring.")
# --- Original Text Message Handler --- END

# --- Voice Message Handler --- START
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles voice messages: downloads, transcribes, and processes the text."""
    user_id = str(update.effective_user.id)
    chat_id = str(update.message.chat.id)
    chat_type = update.message.chat.type
    logger.info(f"Received voice message from user {user_id} in chat {chat_id} (type: {chat_type})")

    if chat_type != "private":
        logger.info("Voice message received in group chat. Replying with instructions.")
        await update.message.reply_text("Для обработки голосовых сообщений о расходах, пожалуйста, отправляйте их мне в личном чате.")
        return

    if not whisper_model:
        await update.message.reply_text("Извините, обработка голосовых сообщений сейчас недоступна (модель ASR не загружена). Пожалуйста, отправьте текстом.")
        return

    voice = update.message.voice
    file_id = voice.file_id
    downloaded_file_path = None
    
    try:
        await update.message.reply_text("🎙️ Обрабатываю голосовое сообщение... Пожалуйста, подождите.")
        voice_file = await context.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".oga") as temp_audio:
            downloaded_file_path = temp_audio.name
            logger.info(f"Downloading voice file to: {downloaded_file_path} (user: {user_id}, chat: {chat_id})")
            await voice_file.download_to_drive(downloaded_file_path)
            logger.info(f"Voice file downloaded successfully (user: {user_id}, chat: {chat_id}).")

        logger.info(f"Transcribing {downloaded_file_path} using model '{WHISPER_MODEL_SIZE}' (user: {user_id}, chat: {chat_id})...")
        segments, info = whisper_model.transcribe(downloaded_file_path, beam_size=5)
        transcript = "".join(segment.text for segment in segments).strip()
        logger.info(f"Transcription result for user {user_id}/chat {chat_id}: {transcript}")
        logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f} (user: {user_id}, chat: {chat_id})")

        if not transcript:
             await update.message.reply_text("Не удалось распознать речь в голосовом сообщении. 😟")
             return
             
        await process_expense_text(transcript, user_id, chat_id, chat_type, update, context)
    except Exception as e:
        logger.error(f"Error processing voice message for user {user_id}/chat {chat_id}: {e}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("Произошла ошибка при обработке вашего голосового сообщения. 😥 Попробуйте еще раз или отправьте текстом.")
    finally:
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try: os.remove(downloaded_file_path); logger.info(f"Cleaned up temporary audio file: {downloaded_file_path}")
            except OSError as e: logger.error(f"Error deleting temporary audio file {downloaded_file_path}: {e}")
# --- Voice Message Handler --- END

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}")
    # Add more detailed logging if context.error provides it
    if hasattr(context.error, '__traceback__'):
         logger.error(traceback.format_exception(None, context.error, context.error.__traceback__))


def main() -> None:
    """Run the bot."""
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Command handlers (as before)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("set_script_url", set_script_url_command))
    application.add_handler(CommandHandler("remove_last_expense", remove_last_expense_command))
    application.add_handler(CommandHandler("correct_last_expense", correct_last_expense_command))

    # Text message handler (now calls refactored function)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # --- Add Voice Message Handler --- START
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    # --- Add Voice Message Handler --- END

    # Error handler (as before)
    application.add_error_handler(error_handler)

    logger.info("Starting bot polling...")
    application.run_polling()

if __name__ == "__main__":
    main() 
