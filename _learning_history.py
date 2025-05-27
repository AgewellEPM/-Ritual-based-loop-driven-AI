# _learning_history.py
import json
import os
from datetime import datetime

class LearningHistory:
    def __init__(self, session_id, max_turns=10):
        self.session_id = session_id
        self.max_turns = max_turns
        self.history = [] # List of complete turn data

    def add_turn(self, turn_data):
        """
        Adds a complete turn's data to the history.
        Turn data should include: question_asked, expert_responses, super_agent_synthesis, etc.
        """
        self.history.append(turn_data)
        # Keep history within max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self):
        """Returns the current learning history."""
        return self.history

    def get_concise_history_for_prompt(self):
        """
        Returns a condensed version of the history suitable for LLM prompts
        to avoid context window limits.
        """
        concise_history = []
        for turn in self.history:
            concise_turn = {
                "turn_number": turn.get("turn_number"),
                "question_asked": turn.get("question_asked", "")[:150] + "...", # Truncate
                "super_agent_synthesis_summary": turn.get("super_agent_synthesis", "")[:250] + "...", # Truncate
                "grade_overall": turn.get("grade_data", {}).get("overall_grade", "N/A"),
                "reflection_summary": turn.get("reflection_data", {}).get("reflection_summary", "")[:100] + "..." # Truncate
            }
            concise_history.append(concise_turn)
        return concise_history
