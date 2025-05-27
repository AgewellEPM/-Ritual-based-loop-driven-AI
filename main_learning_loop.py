# main_learning_loop.py

import os
import json
import time
import logging
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai
# import anthropic # Uncomment if you use Anthropic Claude
# from groq import Groq # Uncomment if you use Groq for Llama/Mixtral
from dotenv import load_dotenv

# Import our modularized components
from _llm_utils import call_llm_with_retry
from _agent_profiles import SUPER_AGENT_PROFILES, EXPERT_AGENT_PROFILES
from _learning_modules import (
    simulate_learning_turn, grade_learning_turn, reflect_on_learning_turn,
    dream_about_topic, collaborate_on_ideas
)
from _learning_history import LearningHistory
from _data_formatter import (
    initialize_session_log, finalize_session_log,
    add_turn_to_session_log, append_training_data_from_turn
)

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize LLM Clients ---
# Initialize OpenAI client for Super Agent (or specific expert roles)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Google Gemini client for expert role
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_generative_model = genai.GenerativeModel('gemini-1.5-flash') # Use the client object

# Initialize other expert clients here if you have them configured
# anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Map client instances to expert profiles
EXPERT_LLM_INSTANCES = {
    "openai_gpt": {"client": openai_client, "model": EXPERT_AGENT_PROFILES["openai_gpt"]["model"]},
    "google_gemini": {"client": gemini_generative_model, "model": EXPERT_AGENT_PROFILES["google_gemini"]["model"]},
    # "grok_expert": {"client": groq_client, "model": EXPERT_AGENT_PROFILES["grok_expert"]["model"]},
    # "claude_expert": {"client": anthropic_client, "model": EXPERT_AGENT_PROFILES["claude_expert"]["model"]},
    # "mistral_expert": {"client": mistral_client_instance, "model": EXPERT_AGENT_PROFILES["mistral_expert"]["model"]},
}

# --- Configuration for the Learning Loop ---
# Default Super Agent Profile Key (can be changed via UI in future)
SELECTED_SUPER_AGENT_PROFILE_KEY = "technical_master"
selected_super_agent_profile = SUPER_AGENT_PROFILES.get(SELECTED_SUPER_AGENT_PROFILE_KEY)
if not selected_super_agent_profile:
    logger.error(f"Super Agent profile '{SELECTED_SUPER_AGENT_PROFILE_KEY}' not found. Exiting.")
    exit()

# Assign a model to the selected super agent profile (e.g., GPT-4o for a powerful SA)
selected_super_agent_profile["model"] = "gpt-4o" # Or "gpt-3.5-turbo-0125" if you prefer
# You would also need to assign the appropriate client instance. For simplicity,
# we'll use openai_client as the "super_agent_client" below. If your SA uses a
# different API (e.g., Gemini-Pro), you'd assign that client here.
super_agent_api_client = openai_client # Assuming SA uses OpenAI

MAX_LEARNING_TURNS = 10 # Limit the number of iterations for a single session
DREAM_INTERVAL = 3    # Super Agent will 'dream' every N turns
COLLAB_INTERVAL = 5   # Super Agent will 'collaborate' every N turns

# --- Main Learning Loop ---
def run_learning_session():
    session_id = f"super_agent_learning_{int(time.time() * 1000)}_{selected_super_agent_profile['profile_name'].replace(' ', '_')}"
    logger.info(f"--- Starting New Super Agent Learning Session: {session_id} ---")
    logger.info(f"Selected Super Agent Profile: {selected_super_agent_profile['profile_name']}")

    initial_topic = input("Enter the topic the Super Agent should learn about and master: ")
    logger.info(f"Super Agent will learn about: '{initial_topic}'")

    # This is where your "prep-rompt" logic would come in for a UI
    # For now, let's use a generic initial question
    current_question_for_experts = f"What are the foundational concepts and key aspects of {initial_topic}?"

    # Initialize learning history
    learning_history = LearningHistory(session_id, max_turns=MAX_LEARNING_TURNS)

    # Initialize the comprehensive session log
    session_log = initialize_session_log(session_id, initial_topic, selected_super_agent_profile.copy())

    for turn_num in range(1, MAX_LEARNING_TURNS + 1):
        logger.info(f"\n--- Learning Turn {turn_num} ---")
        current_knowledge_state = selected_super_agent_profile.get("current_knowledge_state", "novice")
        logger.info(f"Super Agent ({current_knowledge_state}) asks Expert LLMs: '{current_question_for_experts}'")

        # Prepare turn data dictionary *before* execution to capture all details
        current_turn_data = {
            "session_id": session_id,
            "turn_number": turn_num,
            "timestamp_turn_start": datetime.now().isoformat(),
            "initial_topic": initial_topic,
            "super_agent_profile_at_turn": selected_super_agent_profile.copy(), # Snapshot of profile state
            "question_asked": current_question_for_experts,
            "expert_responses": {},
            "super_agent_synthesis": "",
            "next_questions": [], # From synthesis
            "grade_data": {},
            "reflection_data": {},
            "dream_data": {},
            "collaboration_data": {}
        }

        try:
            # 1. Simulate Learning Turn (Query Experts & Initial Synthesis)
            turn_results = simulate_learning_turn(
                super_agent_api_client, # Use the actual client for the SA
                EXPERT_LLM_INSTANCES, # Dictionary of expert clients
                initial_topic,
                current_question_for_experts,
                learning_history.get_concise_history_for_prompt(), # Pass concise history for prompt
                selected_super_agent_profile
            )
            current_turn_data.update({
                "expert_responses": turn_results["expert_responses"],
                "super_agent_synthesis": turn_results["super_agent_synthesis"],
                "next_questions": turn_results["next_questions_for_experts"]
            })

            logger.info("\n--- Super Agent Synthesis ---")
            logger.info(current_turn_data["super_agent_synthesis"])

            # 2. Grade the Turn
            grade_data = grade_learning_turn(
                super_agent_api_client, # Grading is done by the Super Agent's main LLM
                initial_topic,
                current_turn_data["question_asked"],
                current_turn_data["expert_responses"],
                current_turn_data["super_agent_synthesis"],
                current_turn_data["next_questions"],
                learning_history.get_concise_history_for_prompt()
            )
            current_turn_data["grade_data"] = grade_data
            logger.info(f"--- Grade: {grade_data.get('overall_grade', 'N/A'):.2f} (Reason: {grade_data.get('grade_reasoning', 'No reason.')}) ---")

            # 3. Reflect on the Turn
            reflection_data = reflect_on_learning_turn(
                super_agent_api_client, # Reflection by Super Agent's main LLM
                initial_topic,
                current_turn_data,
                grade_data,
                learning_history.get_concise_history_for_prompt()
            )
            current_turn_data["reflection_data"] = reflection_data
            logger.info(f"--- Reflection: {reflection_data.get('reflection_summary', 'No summary.')} ---")

            # Update super agent profile based on reflection
            if "suggested_strategy_adjustments" in reflection_data:
                for key, value in reflection_data["suggested_strategy_adjustments"].items():
                    if key in selected_super_agent_profile:
                        # Handle specific adjustments, e.g., for dreaming_tendency
                        if key == "dreaming_tendency_adjustment":
                            if value == "increase" and selected_super_agent_profile["dreaming_tendency"] == "medium":
                                selected_super_agent_profile["dreaming_tendency"] = "high"
                            elif value == "decrease" and selected_super_agent_profile["dreaming_tendency"] == "medium":
                                selected_super_agent_profile["dreaming_tendency"] = "low"
                            # Add more complex logic if needed
                        else:
                            selected_super_agent_profile[key] = value
                logger.info(f"Super Agent Profile Adjusted: {selected_super_agent_profile}")

            # 4. Dreaming Phase (Conditional)
            if turn_num % DREAM_INTERVAL == 0 and selected_super_agent_profile["dreaming_tendency"] != "low":
                logger.info(f"\n--- Dreaming about '{initial_topic}' ---")
                dream_data = dream_about_topic(
                    super_agent_api_client,
                    initial_topic,
                    current_turn_data["super_agent_synthesis"],
                    learning_history.get_concise_history_for_prompt(),
                    selected_super_agent_profile["dreaming_tendency"]
                )
                current_turn_data["dream_data"] = dream_data
                logger.info(f"Dream Ideas: {dream_data.get('dream_ideas', [])}")

                # Optionally, use a dream idea as the basis for the next question or collaboration
                if dream_data.get("dream_ideas") and turn_num % COLLAB_INTERVAL != 0:
                    # Choose one dream idea to explore
                    current_question_for_experts = f"Considering the dream idea: '{dream_data['dream_ideas'][0]}', how do the foundational concepts of {initial_topic} apply to this, or what new questions does this raise?"
                    logger.info(f"Dreaming led to next question: '{current_question_for_experts}'")

            # 5. Collaboration Phase (Conditional)
            if turn_num % COLLAB_INTERVAL == 0:
                logger.info(f"\n--- Collaborating on '{initial_topic}' ---")
                # Choose an idea to collaborate on
                idea_for_collaboration = (
                    current_turn_data["dream_data"].get("dream_ideas", ["A challenging aspect of " + initial_topic])[0]
                    if current_turn_data["dream_data"].get("dream_ideas") else
                    current_turn_data["super_agent_synthesis"][:100] + "..." # Fallback to part of synthesis
                )
                collaboration_data = collaborate_on_ideas(
                    super_agent_api_client,
                    EXPERT_LLM_INSTANCES,
                    initial_topic,
                    idea_for_collaboration,
                    selected_super_agent_profile["collaboration_style"],
                    EXPERT_AGENT_PROFILES # Pass full expert profiles for their modes
                )
                current_turn_data["collaboration_data"] = collaboration_data
                logger.info(f"Collaboration Result: {collaboration_data.get('summary', 'No summary provided')}")

                # Collaboration might also lead to new questions
                if collaboration_data.get("new_questions"):
                    current_question_for_experts = collaboration_data["new_questions"][0]
                    logger.info(f"Collaboration led to next question: '{current_question_for_experts}'")

            # Finalize turn data timestamp
            current_turn_data["timestamp_turn_end"] = datetime.now().isoformat()

            # Add to session log (for human review) and individual training data files (for ML)
            add_turn_to_session_log(session_log, current_turn_data)
            append_training_data_from_turn(current_turn_data)

            # Add concise turn data to learning history for next iteration's prompt
            learning_history.add_turn({
                "turn_number": current_turn_data["turn_number"],
                "question_asked": current_turn_data["question_asked"],
                "super_agent_synthesis_summary": current_turn_data["super_agent_synthesis"][:250] + "...",
                "grade_overall": current_turn_data["grade_data"].get("overall_grade", "N/A"),
                "reflection_summary": current_turn_data["reflection_data"].get("reflection_summary", "")[:100] + "..."
            })


            # Set next question if not already determined by dreaming/collaboration
            if current_question_for_experts == current_turn_data["question_asked"]: # Check if it wasn't updated
                 if current_turn_data["next_questions"]:
                    current_question_for_experts = current_turn_data["next_questions"][0]
                 else:
                    logger.info("Super Agent has no new questions. Learning session complete.")
                    break

        except ConnectionError as e:
            logger.error(f"API Error during turn {turn_num}: {e}. Ending learning session.", exc_info=True)
            break
        except Exception as e:
            logger.error(f"Unexpected error during turn {turn_num}: {e}. Ending learning session.", exc_info=True)
            break

        time.sleep(2) # Pause between turns for readability

    # Finalize and save the comprehensive session log
    finalize_session_log(session_log, selected_super_agent_profile)
    logger.info(f"--- Super Agent Learning session complete. Log saved to: {session_log['session_id']}.json ---")

if __name__ == "__main__":
    run_learning_session()
