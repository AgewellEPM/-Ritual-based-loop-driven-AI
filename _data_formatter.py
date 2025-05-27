# _data_formatter.py
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Directory for comprehensive session logs (for human review)
SESSION_LOG_DIR = "super_agent_learning_sessions"
os.makedirs(SESSION_LOG_DIR, exist_ok=True)

# Directory for individual JSONL files for specific training datasets (for ML engineers)
TRAINING_DATA_DIR = "training_data_artifacts"
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

TRAINING_DATA_FILES = {
    "synthesis_qa": os.path.join(TRAINING_DATA_DIR, "training_data_synthesis_qa.jsonl"),
    "grade_feedback": os.path.join(TRAINING_DATA_DIR, "training_data_grade_feedback.jsonl"),
    "reflection_feedback": os.path.join(TRAINING_DATA_DIR, "training_data_reflection_feedback.jsonl"),
    "dream_generation": os.path.join(TRAINING_DATA_DIR, "training_data_dream_generation.jsonl"),
    "collaboration_history": os.path.join(TRAINING_DATA_DIR, "training_data_collaboration_history.jsonl")
}

def initialize_session_log(session_id, initial_topic, super_agent_profile):
    """Initializes the main session log structure."""
    return {
        "session_id": session_id,
        "timestamp_start": datetime.now().isoformat(),
        "initial_topic": initial_topic,
        "super_agent_profile_initial": super_agent_profile,
        "turns": []
    }

def finalize_session_log(session_log, super_agent_profile_final):
    """Adds final details and saves the complete session log."""
    session_log["timestamp_end"] = datetime.now().isoformat()
    session_log["super_agent_profile_final"] = super_agent_profile_final
    log_filename = os.path.join(SESSION_LOG_DIR, f"{session_log['session_id']}.json")
    with open(log_filename, "w") as f:
        json.dump(session_log, f, indent=2)
    logger.info(f"Session log saved to: {log_filename}")


def add_turn_to_session_log(session_log, turn_data):
    """Appends a completed turn's data to the session log."""
    session_log["turns"].append(turn_data)
    logger.debug(f"Turn {turn_data['turn_number']} added to session log.")

def append_training_data_from_turn(turn_data):
    """
    Formats key aspects of a completed turn into various JSONL formats
    for specific training datasets.
    """
    metadata = {
        "session_id": turn_data["session_id"],
        "turn_number": turn_data["turn_number"],
        "timestamp": turn_data.get("timestamp_turn_end", datetime.now().isoformat()),
        "initial_topic": turn_data["initial_topic"],
        "super_agent_knowledge_state": turn_data["super_agent_profile_at_turn"].get("current_knowledge_state", "N/A")
    }

    # Synthesis QA
    synthesis_qa_entry = {
        "question": turn_data["question_asked"],
        "expert_responses": turn_data["expert_responses"],
        "super_agent_synthesis": turn_data["super_agent_synthesis"],
        "meta": metadata
    }
    with open(TRAINING_DATA_FILES["synthesis_qa"], "a") as f:
        f.write(json.dumps(synthesis_qa_entry) + "\n")

    # Grade Feedback
    if turn_data.get("grade_data"):
        grade_feedback_entry = {
            "input_context": {
                "question": turn_data["question_asked"],
                "synthesis": turn_data["super_agent_synthesis"],
                "next_questions": turn_data["next_questions"]
            },
            "grade": turn_data["grade_data"],
            "meta": metadata
        }
        with open(TRAINING_DATA_FILES["grade_feedback"], "a") as f:
            f.write(json.dumps(grade_feedback_entry) + "\n")

    # Reflection Feedback
    if turn_data.get("reflection_data"):
        reflection_feedback_entry = {
            "turn_summary": {
                "question": turn_data["question_asked"],
                "synthesis_excerpt": turn_data["super_agent_synthesis"][:200] + "...",
                "grade_overall": turn_data["grade_data"].get("overall_grade") if turn_data.get("grade_data") else "N/A"
            },
            "reflection": turn_data["reflection_data"],
            "meta": metadata
        }
        with open(TRAINING_DATA_FILES["reflection_feedback"], "a") as f:
            f.write(json.dumps(reflection_feedback_entry) + "\n")

    # Dream Generation
    if turn_data.get("dream_data") and turn_data["dream_data"].get("dream_ideas"):
        dream_entry = {
            "topic": turn_data["initial_topic"],
            "current_understanding": turn_data["super_agent_synthesis"],
            "dream_ideas": turn_data["dream_data"].get("dream_ideas"),
            "meta": metadata
        }
        with open(TRAINING_DATA_FILES["dream_generation"], "a") as f:
            f.write(json.dumps(dream_entry) + "\n")

    # Collaboration History
    if turn_data.get("collaboration_data"):
        collab_entry = {
            "topic": turn_data["initial_topic"],
            "initial_idea": turn_data["collaboration_data"].get("initial_idea"),
            "expert_feedback": turn_data["collaboration_data"].get("expert_feedback"),
            "super_agent_refined_idea": turn_data["collaboration_data"].get("refined_idea"),
            "meta": metadata
        }
        with open(TRAINING_DATA_FILES["collaboration_history"], "a") as f:
            f.write(json.dumps(collab_entry) + "\n")
