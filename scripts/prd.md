# Project: Telegram Expense Bot with MCP Integration

## 1. Overview
Develop a Telegram chatbot that allows users to record expenses by chatting with an AI agent. The bot will use a custom MCP (Multi-Capability Platform) server to interact with a Google Sheet, adding expense entries via a Google Apps Script function.

## 2. Core Features
    - **Telegram Bot Interface:** Users interact with the bot via Telegram.
    - **AI Agent Communication:** An AI agent interprets user messages to understand commands, specifically for adding expenses.
    - **Custom MCP Server:** A dedicated MCP server hosts the logic for interacting with external services (Google Sheets).
    - **Expense Recording (`addExpence`):**
        - The MCP server provides an `addExpence` function.
        - This function calls a corresponding `addExpence` function within a Google Apps Script linked to a specified Google Sheet.
    - **Integration:** The AI agent within the Telegram bot can invoke the `addExpence` function on the custom MCP server.

## 3. Technical Requirements
    - **Telegram Bot:** Use a suitable library (e.g., `node-telegram-bot-api` for Node.js, `python-telegram-bot` for Python).
    - **AI Agent:** Integrate with an LLM API (e.g., OpenAI, Anthropic, Gemini). Implement logic to parse user intent (adding expenses) and extract relevant details (amount, category, description).
    - **MCP Server:** Build a server capable of hosting and exposing MCP functions. Define the `addExpence` function signature.
    - **Google Apps Script:** Write a script within the target Google Sheet containing an `addExpence` function that appends a new row with expense data. Securely deploy this script as a web app or API executable.
    - **MCP <-> Google Sheet Communication:** The MCP server needs to make authenticated HTTPS requests to the Google Apps Script endpoint.
    - **Bot <-> MCP Communication:** The Telegram bot needs an MCP client to connect to and call functions on the custom MCP server.

## 4. Non-Functional Requirements
    - **Security:** Ensure API keys and credentials for Telegram, the LLM, and Google Apps Script are handled securely.
    - **Error Handling:** Implement robust error handling for API calls and user interactions.
    - **Deployment:** Define deployment strategies for both the Telegram bot and the MCP server. 
