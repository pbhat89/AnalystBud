# ============================================================================
# Loading key packages
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import langchain as lc
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import anyio
from claude_code_sdk import query
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import seaborn as sns
from langchain.schema import SystemMessage, HumanMessage
import logging
from typing import Any, Optional
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import warnings
import io
import base64
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
import time


# ============================================================================
# Loggings and Warnings
# ============================================================================


# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress INFO logs from httpx and httpcore
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Suppress FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


# ============================================================================
# Env set up
# ============================================================================


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ============================================================================
# Helper functions and classes
# ============================================================================


# Custom Output Parser for Agent Results
class AgentResultParser(BaseOutputParser):
    """Custom parser to handle agent execution results"""

    def parse(self, text: str) -> str:
        """Parse the agent output text"""
        if not text or text.strip() == "":
            raise OutputParserException("Empty or None output received from agent")
        return text.strip()

    @property
    def _type(self) -> str:
        return "agent_result_parser"


def get_agent_variable(agent: Any, variable_name: str) -> Optional[Any]:
    """
    Retrieves a variable from the agent's Python tool local execution environment.
    """
    try:
        python_tool = agent.tools[0]
        return getattr(python_tool, "locals", {}).get(variable_name)
    except (IndexError, AttributeError) as e:
        logging.error(
            f"Could not access tools or local variables from the agent. Details: {e}"
        )
        return None


def format_output_response(result_type: str, raw_result: Any, user_query: str) -> dict:
    """
    Format the raw result into a consistent response structure for the frontend.
    """
    try:
        if raw_result is None:
            return {
                "status": "error",
                "message": f"No result generated for query: {user_query}",
                "data": {
                    "error": f"The agent did not produce any {result_type} result"
                },
                "result_type": result_type,
            }

        if result_type == "statistic":
            return {
                "status": "success",
                "message": f"Statistical analysis complete: {raw_result}",
                "data": {
                    "value": raw_result,
                    "type": "scalar",
                    "description": f"Result for: {user_query}",
                },
                "result_type": "statistic",
            }

        elif result_type == "table":
            if hasattr(raw_result, "to_dict"):
                return {
                    "status": "success",
                    "message": f"Data table generated successfully with {len(raw_result)} rows and {len(raw_result.columns)} columns",
                    "data": {
                        "dataframe": raw_result,
                        "type": "dataframe",
                        "shape": raw_result.shape,
                        "columns": list(raw_result.columns),
                        "description": f"Table result for: {user_query}",
                    },
                    "result_type": "table",
                }
            else:
                return {
                    "status": "error",
                    "message": "Expected a DataFrame but received a different data type",
                    "data": {
                        "error": f"Received {type(raw_result)} instead of DataFrame"
                    },
                    "result_type": "table",
                }

        elif result_type == "graph":
            if raw_result is not None:
                if hasattr(raw_result, "savefig"):
                    return {
                        "status": "success",
                        "message": "Visualization generated successfully",
                        "data": {
                            "figure": raw_result,
                            "type": "matplotlib",
                            "description": f"Graph result for: {user_query}",
                        },
                        "result_type": "graph",
                    }
                elif hasattr(raw_result, "show") and hasattr(raw_result, "data"):
                    return {
                        "status": "success",
                        "message": "Interactive visualization generated successfully",
                        "data": {
                            "figure": raw_result,
                            "type": "plotly",
                            "description": f"Interactive graph result for: {user_query}",
                        },
                        "result_type": "graph",
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Generated figure is not in a recognized format",
                        "data": {
                            "error": f"Figure type {type(raw_result)} not supported"
                        },
                        "result_type": "graph",
                    }
            else:
                return {
                    "status": "error",
                    "message": "No visualization was generated",
                    "data": {"error": "The agent did not produce a figure"},
                    "result_type": "graph",
                }

        else:
            return {
                "status": "error",
                "message": f"Unknown result type: {result_type}",
                "data": {"error": f"Unsupported result type: {result_type}"},
                "result_type": result_type,
            }

    except Exception as e:
        logging.error(f"Error formatting output: {e}")
        return {
            "status": "error",
            "message": f"Error processing {result_type} result: {str(e)}",
            "data": {"error": str(e)},
            "result_type": result_type,
        }


# ============================================================================
# Main Function
# ============================================================================


def agent_at_work(model: Any, user_query: str, path: Any) -> dict:
    """
    Invokes a pandas dataframe agent to process a user query, classifying the expected output
    and returning a formatted result for the frontend.
    Enhanced with OutputFixingParser for robust error handling.
    """

    # =============================================
    # # Define classification prompt as a constant
    # ==============================================

    CLASSIFICATION_PROMPT = """
    You are a classifier. Read the user request carefully and decide what type of output it requires.
    There are ONLY three possible categories: `statistic`, `table`, or `graph`.
    RULES:
    - The answer MUST be strictly one word: either `statistic`, `table`, or `graph`.
    - Do not explain. Do not output anything else.
    - Example: User request: "generate a bar plot of gender" -> Answer: graph
    - Example: User request: "show me the correlation matrix" -> Answer: table
    - Example: User request: "what is the average age" -> Answer: statistic
    """

    try:

        # =============================================
        # Step 0: Define LLM
        # ==============================================
        def create_chat_openai(model):
            """Create a ChatOpenAI instance with a specified model."""
            try:
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=OPENROUTER_API_KEY,
                    openai_api_base=OPENROUTER_BASE_URL,
                    streaming=False,
                    verbose=False,
                    temperature=0.0,
                    max_retries=3,
                )
                return llm
            except Exception as e:
                logging.error(f"Failed to create ChatOpenAI instance: {e}")
                return None

        llm = create_chat_openai(model)
        if llm is None:
            return {
                "status": "error",
                "message": "Failed to initialize the language model",
                "data": {"error": "LLM initialization failed"},
                "result_type": "unknown",
            }

        # =============================================
        # Step 1: Create custom output parser and wrap it with OutputFixingParser
        # ==============================================

        base_parser = AgentResultParser()
        output_fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

        #
        # =============================================
        # Step 2: Classify the user query with error handling
        # ==============================================
        try:
            classification_result = llm.invoke(
                CLASSIFICATION_PROMPT + f"\nUser request: '{user_query}'"
            )
            prompt_type = classification_result.content.strip().lower()

            # NEW: Use OutputFixingParser to ensure valid classification
            try:
                fixed_classification = output_fixing_parser.parse(prompt_type)
                prompt_type = fixed_classification.lower()
            except OutputParserException as parse_error:
                logging.warning(
                    f"Classification parsing failed: {parse_error}. Using fallback."
                )
                prompt_type = "table"  # Fallback to table

        except Exception as e:
            logging.error(f"LLM classification failed: {e}")
            return {
                "status": "error",
                "message": "Failed to classify the user query",
                "data": {"error": str(e)},
                "result_type": "unknown",
            }

        valid_types = {"statistic", "table", "graph"}
        if prompt_type not in valid_types:
            logging.warning(
                f"Invalid classification type returned: '{prompt_type}'. Defaulting to 'table'."
            )
            prompt_type = "table"

        # =============================================
        # Step 3:  Formulate the final prompt based on the classification
        # ==============================================
        prompt_map = {
            "statistic": f"{user_query} Assign the final result value to a variable named 'sg_value'.",
            "table": f"{user_query} Assign the output dataframe to a variable named 'temp_df'.",
            "graph": f"{user_query} Create the visualization and assign the figure object to a variable named 'figure'.",
        }
        final_prompt = prompt_map.get(prompt_type, user_query)

        # Load the dataframe
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load CSV file: {str(e)}",
                "data": {"error": f"File loading error: {str(e)}"},
                "result_type": prompt_type,
            }

        #
        # =============================================
        # Step 4:  Create the agent
        # ==============================================
        try:
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True,
                return_intermediate_steps=False,
            )
        except Exception as e:
            logging.error(f"Agent creation failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to create the analysis agent: {str(e)}",
                "data": {"error": str(e)},
                "result_type": prompt_type,
            }

        # =============================================
        # Step 5:  Enhanced agent invocation with retry mechanism
        # ==============================================

        max_retries = 10  # can be adjusted but in my experience , successful result is usually obtained within 3-5 tries
        retry_count = 0
        raw_result = None

        while retry_count < max_retries and raw_result is None:
            try:
                # Invoke the agent
                agent_response = agent.invoke({"input": final_prompt})

                # NEW: Apply OutputFixingParser to the agent response if it's a string
                if isinstance(agent_response, str):
                    try:
                        fixed_response = output_fixing_parser.parse(agent_response)
                        logging.info(
                            f"Agent response fixed by OutputFixingParser: {fixed_response}"
                        )
                    except OutputParserException as parse_error:
                        logging.warning(
                            f"OutputFixingParser failed: {parse_error}. Using original response."
                        )

                # Retrieve the result from agent's local variables
                variable_map = {
                    "statistic": "sg_value",
                    "table": "temp_df",
                    "graph": "figure",
                }
                variable_name = variable_map.get(prompt_type)
                raw_result = get_agent_variable(agent, variable_name)

                # NEW: If result is still None, wait and retry
                if raw_result is None:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(
                            f"Attempt {retry_count} failed to get result. Retrying..."
                        )
                        time.sleep(1)  # Brief pause before retry
                    else:
                        logging.error(
                            f"Failed to get result after {max_retries} attempts"
                        )

            except OutputParserException as parse_error:
                logging.error(
                    f"Output parsing error on attempt {retry_count + 1}: {parse_error}"
                )
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying agent invocation due to parsing error...")
                    time.sleep(1)
                else:
                    return {
                        "status": "error",
                        "message": f"Output parsing failed after {max_retries} attempts: {str(parse_error)}",
                        "data": {"error": str(parse_error)},
                        "result_type": prompt_type,
                    }

            except Exception as e:
                logging.error(
                    f"Agent invocation failed on attempt {retry_count + 1}: {e}"
                )
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying agent invocation due to general error...")
                    time.sleep(1)
                else:
                    return {
                        "status": "error",
                        "message": f"The agent could not process the query after {max_retries} attempts: {str(e)}",
                        "data": {"error": str(e)},
                        "result_type": prompt_type,
                    }

        # =============================================
        # Step 5:  Final validation and formatting & last attempts
        # ==============================================

        if raw_result is None:
            #  Last attempt with OutputFixingParser to generate a meaningful error response
            try:
                error_prompt = f"The agent failed to generate {prompt_type} for query: {user_query}. Provide a brief explanation."
                error_response = llm.invoke(error_prompt)
                fixed_error = output_fixing_parser.parse(error_response.content)
                return {
                    "status": "error",
                    "message": f"No result generated after all retry attempts: {fixed_error}",
                    "data": {"error": "Maximum retry attempts exceeded"},
                    "result_type": prompt_type,
                }
            except:
                return {
                    "status": "error",
                    "message": f"No result generated after {max_retries} retry attempts",
                    "data": {"error": "Maximum retry attempts exceeded"},
                    "result_type": prompt_type,
                }

        # Format and return the response
        return format_output_response(prompt_type, raw_result, user_query)

    except Exception as e:
        logging.error(f"Unexpected error in agent_at_work: {e}")
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
            "data": {"error": str(e)},
            "result_type": "unknown",
        }
