import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import os
# from datetime import date # Keep commented out
from typing import List, Any, Dict
import httpx # Ensure httpx is imported

from dotenv import load_dotenv

# Determine the root directory of the project
# Assuming main.py is in src/mcp_server, the root is two levels up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')

# Explicitly load .env from the project root
if os.path.exists(DOTENV_PATH):
    load_dotenv(dotenv_path=DOTENV_PATH)
    print(f"INFO: Loading .env file from: {DOTENV_PATH}") # Add print for confirmation
else:
    print(f"WARNING: .env file not found at: {DOTENV_PATH}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Expense Bot MCP Server",
    description="Handles function calls for the Telegram Expense Bot by POSTing to Google Apps Script Web App URL.",
    version="0.1.0"
)

# --- Argument Models --- #
class AddExpenseArgsGAS(BaseModel):
    # Model now needs to implicitly handle script_url passed in args, 
    # but validation only applies to actual expense fields.
    reason: str = Field(..., min_length=1, max_length=200, description="Reason/description for the expense")
    amount: float = Field(..., gt=0, description="The expense amount, must be positive")
    spenderIndex: int = Field(..., ge=0, description="Index of the participant who spent the money")
    categoryIndex: int = Field(..., ge=0, description="Index of the expense category")
    coefficients: List[Any] # Allow numbers or empty strings initially
    # script_url is not part of the GAS payload, so not defined here directly

# --- Helper Function for GAS Call --- #
async def call_gas_webapp(script_url: str, function_name: str, payload: dict) -> dict:
    """Helper function to POST to the Google Apps Script Web App URL."""
    # Removed access to app.state, uses passed script_url
    if not script_url:
        logger.error("Cannot call GAS Web App: Script URL was not provided to the helper.")
        return {"status": "error", "message": "Server configuration error: GAS URL missing in request."}

    request_body = {
        "functionName": function_name,
        "payload": payload
    }
    
    async with httpx.AsyncClient() as client:
        logger.info(f"Calling GAS Web App ({function_name}) at {script_url} with body: {request_body}")
        try:
            # Use POST for Web Apps, follow redirects which GAS often uses
            response = await client.post(script_url, json=request_body, timeout=20.0, follow_redirects=True)
            response.raise_for_status() # Raises exception for 4xx/5xx responses

            # Handle functions that might return empty on success
            if function_name in ["addExpense", "removeLastExpense"] and not response.text.strip():
                success_detail = "Operation successful (empty response from GAS)."
                if function_name == "addExpense": success_detail = "Expense added (empty response from GAS)."
                if function_name == "removeLastExpense": success_detail = "Last expense removed (empty response from GAS)."
                logger.info(f"GAS Web App returned empty response for {function_name} (assuming success).")
                return {"status": "success", "details": success_detail}
            
            # Attempt to parse JSON for all other cases or if addExpense returned non-empty
            try: 
                gas_response_data = response.json()
                logger.info(f"GAS Web App Response ({response.status_code}): {gas_response_data}")
                # Assume GAS returns {"status": "success", ...} or {"status": "error", ...} or just the data array
                # Let's try to standardize: if it's a list, wrap it
                if isinstance(gas_response_data, list):
                    # Determine key based on function name for wrapping
                    data_key = "result" # Default key
                    if function_name == "getParticipantsForUrl": data_key = "participants"
                    if function_name == "getCategoriesForUrl": data_key = "categories"
                    return {"status": "success", data_key: gas_response_data}
                elif isinstance(gas_response_data, dict) and "status" not in gas_response_data:
                     # If it's a dict without status, assume success and wrap it
                     return {"status": "success", "result": gas_response_data}
                # Otherwise, return the dict as is (hopefully has status)
                return gas_response_data 
            except Exception as json_error:
                logger.warning(f"Failed to parse JSON response from GAS Web App. Status: {response.status_code}. Content: {response.text[:500]}")
                # If it wasn't addExpense expecting empty, this is likely an error
                if function_name != "addExpense":
                     return {"status": "error", "error_type": "Invalid Response", "details": "Received non-JSON response from Google Apps Script.", "raw_response": response.text[:500]}
                else: # Should have been caught by the empty check above, but just in case
                     return {"status": "warning", "details": "Received non-JSON response from Google Apps Script for addExpense.", "raw_response": response.text[:500]}

        except httpx.TimeoutException:
            logger.error(f"Timeout calling Google Apps Script ({function_name}) at {script_url}")
            return {"status": "error", "error_type": "Timeout Error", "details": "Request to Google Apps Script timed out."}
        except httpx.RequestError as exc:
            logger.error(f"HTTP Request error calling Google Apps Script ({function_name}): {exc}")
            return {"status": "error", "error_type": "Connection Error", "details": f"Could not connect to Google Apps Script: {exc}"}
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP Status error from Google Apps Script ({function_name}): {exc.response.status_code} - {exc.response.text[:500]}")
            # Specific checks for common errors
            if "Script function not found: doPost" in exc.response.text:
                 return {"status": "error", "error_type": "Configuration Error", "details": "Apps Script is missing the doPost(e) function."}
            if "Authorization is required to perform that action" in exc.response.text:
                 return {"status": "error", "error_type": "Authorization Error", "details": "Apps Script authorization failed. Check deployment permissions ('Anyone' or 'Anyone, even anonymous')."}
            return {"status": "error", "error_type": "HTTP Error", "details": f"Google Apps Script returned error {exc.response.status_code}"}
        except Exception as e:
             logger.exception(f"Unexpected error during call_gas_webapp ({function_name}): {e}")
             return {"status": "error", "error_type": "Unexpected Error", "details": str(e)}

# --- MCP Functions --- #
async def add_expense(args: dict) -> dict:
    """Validates expense arguments and calls the GAS Web App using the provided script_url."""
    logger.info(f"Executing MCP add_expense with raw args: {args}")
    script_url = args.get("script_url")
    if not script_url:
        return {"status": "error", "message": "Missing 'script_url' in arguments for add_expense."}
        
    try:
        # Validate only the expense-related fields, ignore script_url for validation
        expense_data_to_validate = {k: v for k, v in args.items() if k != 'script_url'}
        validated_args = AddExpenseArgsGAS(**expense_data_to_validate)
        logger.info(f"Validated add_expense args: {validated_args.model_dump_json()}")
        # Call helper with validated args as payload and the extracted script_url
        gas_result = await call_gas_webapp(script_url=script_url, function_name="addExpense", payload=validated_args.model_dump())
        return gas_result 
    except ValidationError as e:
        logger.error(f"Validation Error for add_expense: {e}")
        return {"status": "error", "error_type": "Validation Error", "details": e.errors()}
    except Exception as e:
        logger.exception(f"Unexpected error during add_expense preparation: {e}")
        return {"status": "error", "error_type": "Unexpected Error", "details": str(e)}

async def get_participants_for_url(args: dict) -> dict:
    """Calls the GAS Web App to get participants using the provided script_url."""
    logger.info(f"Executing MCP get_participants_for_url with args: {args}")
    script_url = args.get("script_url")
    if not script_url:
        return {"status": "error", "message": "Missing 'script_url' in arguments."}
        
    # Call helper with empty payload, passing the script_url
    gas_result = await call_gas_webapp(script_url=script_url, function_name="getParticipants", payload={})
    # Adapt based on actual GAS return structure
    if isinstance(gas_result, dict) and gas_result.get("status") == "success":
         actual_list = gas_result.get("participants")
         if actual_list is None:
              actual_list = gas_result.get("result") # Try generic key
         
         if isinstance(actual_list, list):
             # Flatten if necessary (e.g., [['Name1'], ['Name2']] -> ['Name1', 'Name2'])
             if all(isinstance(item, list) and len(item) == 1 for item in actual_list):
                 actual_list = [item[0] for item in actual_list]
             return {"status": "success", "participants": actual_list}
         else:
             logger.error(f"get_participants_for_url GAS call succeeded but data is not a list or in expected format: {gas_result}")
             return {"status": "error", "error_type": "Unexpected Format", "details": "Received unexpected data format for participants."}
    # If GAS directly returned a list (common for simple gets)
    elif isinstance(gas_result, list):
        if all(isinstance(item, list) and len(item) == 1 for item in gas_result):
             gas_result = [item[0] for item in gas_result]
        return {"status": "success", "participants": gas_result}
    else: # Return error/warning from call_gas_webapp
        return gas_result 

async def get_categories_for_url(args: dict) -> dict:
    """Calls the GAS Web App to get categories using the provided script_url."""
    logger.info(f"Executing MCP get_categories_for_url with args: {args}")
    script_url = args.get("script_url")
    if not script_url:
        return {"status": "error", "message": "Missing 'script_url' in arguments."}
        
    # Call helper with empty payload, passing the script_url
    gas_result = await call_gas_webapp(script_url=script_url, function_name="getCategories", payload={})
    # Adapt based on actual GAS return structure
    if isinstance(gas_result, dict) and gas_result.get("status") == "success":
         actual_list = gas_result.get("categories")
         if actual_list is None:
             actual_list = gas_result.get("result") # Try generic key

         if isinstance(actual_list, list):
             # Flatten if necessary
             if all(isinstance(item, list) and len(item) == 1 for item in actual_list):
                 actual_list = [item[0] for item in actual_list]
             return {"status": "success", "categories": actual_list}
         else:
             logger.error(f"get_categories_for_url GAS call succeeded but data is not a list or in expected dict format: {gas_result}")
             return {"status": "error", "error_type": "Unexpected Format", "details": "Received unexpected data format for categories."}
    # If GAS directly returned a list
    elif isinstance(gas_result, list):
        if all(isinstance(item, list) and len(item) == 1 for item in gas_result):
             gas_result = [item[0] for item in gas_result]
        return {"status": "success", "categories": gas_result}
    else: # Return error/warning from call_gas_webapp
        return gas_result

# --- New Function: Remove Last Expense --- START
async def remove_last_expense(args: dict) -> dict:
    """Calls the removeLastExpense function in the Google Apps Script."""
    script_url = args.get("script_url")
    if not script_url:
        return {"status": "error", "error_type": "Argument Error", "message": "Missing 'script_url' in arguments for removeLastExpense."}
    
    logger.info(f"Calling removeLastExpense on GAS: {script_url}")
    gas_payload = {}
    
    try:
        gas_result = await call_gas_webapp(script_url, "removeLastExpense", gas_payload)
        
        if isinstance(gas_result, dict):
            if gas_result.get("status") == "success":
                logger.info(f"GAS removeLastExpense successful for {script_url}. Result: {gas_result.get('details')}")
                return {"status": "success", "details": gas_result.get('details', "Last expense removed.")}
            else:
                error_detail = gas_result.get("details", "Unknown error from GAS during removal.")
                logger.error(f"GAS removeLastExpense failed for {script_url}: {error_detail}")
                return {"status": "error", "error_type": gas_result.get("error_type", "GAS Error"), "message": error_detail}
        else:
            logger.error(f"Unexpected response type from call_gas_webapp for removeLastExpense ({script_url}): {type(gas_result)}")
            return {"status": "error", "error_type": "Unexpected Response", "message": "Unexpected response format from GAS helper."}
            
    except Exception as e:
        logger.exception(f"Unexpected error in remove_last_expense calling GAS ({script_url}): {e}")
        return {"status": "error", "error_type": "Unexpected Error", "message": f"Unexpected server error calling GAS: {e}"}
# --- New Function: Remove Last Expense --- END

# --- MCP Function Registry --- #
AVAILABLE_FUNCTIONS = {
    "addExpense": add_expense,
    "getParticipantsForUrl": get_participants_for_url,
    "getCategoriesForUrl": get_categories_for_url,
    "removeLastExpense": remove_last_expense
}

# --- Models for API --- #
class FunctionCallRequest(BaseModel):
    function_name: str
    arguments: dict # Note: MCP server expects args under 'arguments', but GAS call payload structure is different

class FunctionCallResponse(BaseModel):
    result: Any | None = None
    error: str | None = None

# --- API Endpoint --- #
@app.post("/call", response_model=FunctionCallResponse)
async def handle_function_call(request: FunctionCallRequest):
    """Handles incoming function call requests from clients like the bot."""
    function_name = request.function_name
    arguments = request.arguments # Arguments from the client (e.g., Telegram bot)
    logger.info(f"Received call for function: {function_name} with args: {arguments}")

    if function_name not in AVAILABLE_FUNCTIONS:
        logger.error(f"Function '{function_name}' not found.")
        # Return structured error in response body
        return FunctionCallResponse(error=f"Function '{function_name}' not implemented")

    function_to_call = AVAILABLE_FUNCTIONS[function_name]

    try:
        # Pass the arguments from the request to the target MCP function
        result_data = await function_to_call(arguments)

        # Check the structured result from the MCP function (which wrapped the GAS call)
        if isinstance(result_data, dict) and result_data.get("status") == "error":
            logger.error(f"MCP function '{function_name}' failed: {result_data.get('message')}")
            # Propagate the error details from the failed call
            return FunctionCallResponse(error=f"{result_data.get('error_type', 'Function Error')}: {result_data.get('message', 'Unknown error')}")
        else:
            # If success or warning, return the whole result dict from the MCP function
            logger.info(f"MCP function '{function_name}' executed successfully.")
            # The bot expects the actual data (e.g., list of participants) in the 'result' field
            # So, we return the dictionary received from the MCP function directly
            return FunctionCallResponse(result=result_data)

    except Exception as e:
        logger.exception(f"Unexpected error during MCP function execution '{function_name}': {e}")
        return FunctionCallResponse(error=f"Unexpected server error: {str(e)}")

# --- Basic Health Check --- #
@app.get("/")
async def root():
    return {"message": "MCP Server is running"}

# --- Run Server --- #
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    logger.info(f"Starting MCP server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True) # reload=True for development 
