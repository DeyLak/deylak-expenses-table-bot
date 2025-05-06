# Telegram Expense Bot

A Telegram bot that uses an AI agent and a custom MCP server to record expenses to a Google Sheet.

## Setup

1.  Clone the repository.
2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file from `.env.example` and fill in the required API keys and configuration.
5.  Run the bot/server (instructions TBD).

## Project Structure

-   `/src`: Main source code
    -   `/src/bot`: Telegram bot logic
    -   `/src/mcp_server`: Custom MCP server logic
-   `/config`: Configuration files
-   `/docs`: Project documentation
-   `/scripts`: Utility scripts (e.g., Google Apps Script)
-   `/tasks`: Task definitions (managed by Task Master)
