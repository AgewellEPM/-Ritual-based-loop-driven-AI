# _learning_modules.py
import json
import logging
from datetime import datetime
from _llm_utils import call_llm_with_retry # Our generalized utility

logger = logging.getLogger(__name__)

def _format_learning_history_for_prompt(learning_history_data):
    """Formats the learning history for inclusion in LLM prompts."""
    if not learning_history_data:
        return ""
    
    formatted_history = "\n--- Recent Learning History (Summarized) ---\n"
    for turn in learning_history_data[-3:]: # Limit to last 3 turns for prompt context
        formatted_history += f"Turn {turn['turn_number']}:\n"
        formatted_history += f"  Q: {turn['question_asked']}\n"
        formatted_history += f"  SA Synthesis: {turn['super_agent_synthesis_summary']}\n"
        formatted_history += f"  Grade: {turn['grade_overall']}\n"
        formatted_history += f"  Reflection: {turn['reflection_summary']}\n"
    formatted_history += "--- End History ---\n\n"
    return formatted_history

def simulate_learning_turn(super_agent_client, expert_llm_clients, topic, current_question, learning_history_for_prompt, super_agent_profile):
    """
    Orchestrates the querying of expert LLMs and the initial synthesis by the super agent.
    """
    expert_responses = {}
    formatted_history = _format_learning_history_for_prompt(learning_history_for_prompt)

    logger.info(f"Querying expert LLMs for topic: '{topic}' with question: '{current_question}'")
    for expert_name, config in expert_llm_clients.items():
        expert_client = config["client"]
        expert_model = config["model"]
        expert_role = config["profile_name"] # Using profile_name as role

        prompt = f"""
        You are {expert_role}, an expert on the topic of "{topic}".
        Your role is to provide {config['role']} for the Super Agent.

        {formatted_history}

        The Super Agent is currently learning about "{topic}" and has asked the following question:
        "{current_question}"

        Please provide a concise and informative response from your specialized perspective.
        """
        try:
            response = call_llm_with_retry(
                expert_client,
                expert_model,
                messages=[
                    {"role": "system", "content": prompt.strip()},
                    {"role": "user", "content": current_question} # The specific question part
                ],
                temperature=0.7,
                max_tokens=500
            )
            expert_responses[expert_name] = response.choices[0].message.content
            logger.debug(f"Received response from {expert_name}")
        except Exception as e:
            logger.error(f"Error getting response from {expert_name}: {e}")
            expert_responses[expert_name] = f"Error: Could not get response from {expert_name}."

    # Super Agent Synthesis
    super_agent_synthesis_prompt = f"""
    You are the Super Agent: {super_agent_profile['profile_name']}.
    Your learning style is {super_agent_profile['learning_style']}.
    You are currently learning about the topic: "{topic}".

    {formatted_history}

    You have asked the following question to your expert advisors:
    "{current_question}"

    Here are their responses:
    {json.dumps(expert_responses, indent=2)}

    Please synthesize these responses into a coherent understanding of the question.
    Identify any consensus, contradictions, unique insights, and remaining gaps.
    Based on this synthesis and your current understanding, formulate 1-2 new, deeper, and relevant questions
    to continue your learning journey on "{topic}".

    Return ONLY a JSON object like this:
    {{
      "synthesis": "...",
      "new_questions": ["...", "..."]
    }}
    """
    try:
        super_agent_response = call_llm_with_retry(
            super_agent_client,
            super_agent_profile["model"], # Super agent uses its own assigned model
            messages=[
                {"role": "system", "content": super_agent_synthesis_prompt.strip()},
                {"role": "user", "content": f"Synthesize and generate next questions for: '{current_question}'"}
            ],
            temperature=0.6,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        content = json.loads(super_agent_response.choices[0].message.content)
        synthesis = content.get("synthesis", "No synthesis provided.")
        next_questions = content.get("new_questions", [])
        logger.debug("Super Agent synthesis complete.")
    except Exception as e:
        logger.error(f"Error during Super Agent synthesis: {e}")
        synthesis = f"Error during synthesis: {e}"
        next_questions = [f"What went wrong during synthesis on {topic}?"]

    return {
        "expert_responses": expert_responses,
        "super_agent_synthesis": synthesis,
        "next_questions_for_experts": next_questions
    }

def grade_learning_turn(super_agent_client, topic, current_question, expert_responses, super_agent_synthesis, next_questions, learning_history_for_prompt):
    """
    Quantitatively assesses the quality of the super agent's understanding, synthesis, and generated questions.
    """
    formatted_history = _format_learning_history_for_prompt(learning_history_for_prompt)

    prompt = f"""
    You are a dedicated grader for the Super Agent's learning process.
    Your task is to evaluate the Super Agent's performance on the following turn related to "{topic}".

    {formatted_history}

    Original Question asked by Super Agent: "{current_question}"
    Expert Responses: {json.dumps(expert_responses, indent=2)}
    Super Agent's Synthesis: "{super_agent_synthesis}"
    Super Agent's Next Questions: {json.dumps(next_questions)}

    Rate the following criteria on a scale of 0.0 to 1.0, and provide a brief reasoning:
    1.  **Relevance (to original question & topic)**: How well did the synthesis address the original question and the overall topic?
    2.  **Coherence & Clarity**: Is the synthesis logical, well-structured, and easy to understand?
    3.  **Completeness (based on expert responses)**: Did the synthesis adequately incorporate information from all expert responses, noting contradictions/agreements?
    4.  **Depth of Understanding**: Does the synthesis demonstrate a deeper understanding compared to merely summarizing?
    5.  **Novelty & Quality of Next Questions**: Are the next questions truly deeper, innovative, and likely to advance the learning?

    Calculate an `overall_grade` as an average or weighted average of the above scores.

    Return ONLY a JSON object:
    {{
      "relevance_score": 0.0,
      "coherence_score": 0.0,
      "completeness_score": 0.0,
      "depth_score": 0.0,
      "novelty_questions_score": 0.0,
      "overall_grade": 0.0,
      "grade_reasoning": "A brief explanation of the overall grade."
    }}
    """
    logger.info("Grading Super Agent's turn...")
    try:
        response = call_llm_with_retry(
            super_agent_client, # Super agent grades itself, or use a dedicated grader LLM
            "gpt-3.5-turbo-0125", # A standard, reliable model for grading
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1, # Keep it deterministic for grading
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        grade_data = json.loads(response.choices[0].message.content)
        # Ensure scores are floats
        for key in grade_data:
            if "_score" in key and isinstance(grade_data[key], (int, float)):
                grade_data[key] = float(grade_data[key])
        return grade_data
    except Exception as e:
        logger.error(f"Error during grading: {e}")
        return {
            "relevance_score": 0.0, "coherence_score": 0.0, "completeness_score": 0.0,
            "depth_score": 0.0, "novelty_questions_score": 0.0, "overall_grade": 0.0,
            "grade_reasoning": f"Grading failed: {e}"
        }

def reflect_on_learning_turn(super_agent_client, topic, turn_data, grade_data, learning_history_for_prompt):
    """
    Qualitatively analyzes the learning process, identifying strengths, weaknesses, and potential improvements.
    """
    formatted_history = _format_learning_history_for_prompt(learning_history_for_prompt)
    
    prompt = f"""
    You are the Super Agent's internal reflection module.
    Analyze the learning process for the current turn on topic "{topic}", considering the grade received.

    {formatted_history}

    Turn Details:
    Question Asked: "{turn_data['question_asked']}"
    Super Agent Synthesis: "{turn_data['super_agent_synthesis']}"
    Grade Data: {json.dumps(grade_data, indent=2)}
    Expert Responses: {json.dumps(turn_data['expert_responses'], indent=2)}

    Based on this, provide a concise reflection:
    1.  What went well in this learning turn?
    2.  What were the primary challenges or areas for improvement?
    3.  Suggest specific adjustments to the Super Agent's learning strategy, dreaming tendency, or collaboration approach for future turns to improve performance. (e.g., "focus more on X," "try a different expert," "increase dreaming," "be more critical")

    Return ONLY a JSON object:
    {{
      "reflection_summary": "A brief summary of the reflection.",
      "areas_for_improvement": ["...", "..."],
      "suggested_strategy_adjustments": {{
        "learning_style_adjustment": "...",
        "dreaming_tendency_adjustment": "increase" | "decrease" | "maintain",
        "collaboration_style_adjustment": "..."
      }}
    }}
    """
    logger.info("Reflecting on Super Agent's turn...")
    try:
        response = call_llm_with_retry(
            super_agent_client,
            "gpt-3.5-turbo-0125", # A standard, reliable model for reflection
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3, # Allow some creativity but keep it grounded
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error during reflection: {e}")
        return {
            "reflection_summary": f"Reflection failed: {e}",
            "areas_for_improvement": [],
            "suggested_strategy_adjustments": {}
        }

def dream_about_topic(super_agent_client, topic, current_understanding_summary, learning_history_for_prompt, dreaming_tendency):
    """
    Encourages the super agent to generate novel ideas, hypothetical scenarios, or future implications.
    """
    if dreaming_tendency == "low":
        logger.info("Dreaming tendency is low, skipping dreaming phase.")
        return {"dream_ideas": [], "dream_summary": "Dreaming tendency low."}

    formatted_history = _format_learning_history_for_prompt(learning_history_for_prompt)

    prompt = f"""
    You are the Super Agent. Your current dreaming tendency is {dreaming_tendency}.
    Based on your current understanding of "{topic}":
    "{current_understanding_summary}"

    {formatted_history}

    Generate 3-5 entirely new, speculative, or creative ideas, applications, challenges, or future directions related to this topic.
    Think broadly and interdisciplinarily. Don't just summarize; extrapolate, hypothesize, or imagine.

    Return ONLY a JSON object:
    {{
      "dream_ideas": [
        "Idea 1: ...",
        "Idea 2: ...",
        "..."
      ],
      "dream_summary": "A brief summary of the ideas generated."
    }}
    """
    logger.info("Super Agent is dreaming...")
    try:
        response = call_llm_with_retry(
            super_agent_client,
            "gpt-3.5-turbo-0125", # Consider a more creative model if available (e.g., gemini-1.5-pro or gpt-4o)
            messages=[{"role": "system", "content": prompt}],
            temperature=0.9, # High temperature for creativity
            max_tokens=600,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error during dreaming: {e}")
        return {"dream_ideas": [], "dream_summary": f"Dreaming failed: {e}"}

def collaborate_on_ideas(super_agent_client, expert_llm_clients, topic, initial_idea, collaboration_style, expert_profiles):
    """
    Facilitates a collaborative brainstorming/refinement session between the super agent and selected expert LLMs.
    """
    logger.info(f"Super Agent initiating collaboration on idea: '{initial_idea}'")

    expert_feedback = {}
    collaboration_log = []

    # Super Agent sets the collaboration context
    sa_collaboration_prompt = f"""
    You are the Super Agent. You are initiating a collaborative discussion on the topic "{topic}".
    The idea we are exploring is: "{initial_idea}"
    Your collaboration style is: {collaboration_style}.

    I will share this idea with the expert advisors. Please prepare to synthesize their feedback
    and refine the idea or formulate new questions based on their input.
    """
    # Simulate initial prompt from SA to experts (no direct API call, just setting the stage)
    collaboration_log.append({"speaker": "Super Agent (Initiator)", "message": sa_collaboration_prompt})

    # Experts provide feedback based on their collaboration mode
    for expert_name, config in expert_llm_clients.items():
        expert_client = config["client"]
        expert_model = config["model"]
        expert_profile = expert_profiles.get(expert_name, {}) # Get full profile for collaboration mode
        expert_collaboration_mode = expert_profile.get("collaboration_mode", "general feedback")

        expert_prompt = f"""
        You are {expert_profile.get('profile_name', expert_name)}. Your role is {expert_profile.get('role', 'an expert')}.
        The Super Agent has proposed an idea for collaboration on the topic "{topic}":
        "{initial_idea}"

        Your collaboration mode is: "{expert_collaboration_mode}".
        Please provide your feedback, critique, alternative perspectives, or suggestions for refinement based on your expertise and collaboration mode.
        """
        try:
            response = call_llm_with_retry(
                expert_client,
                expert_model,
                messages=[
                    {"role": "system", "content": expert_prompt},
                    {"role": "user", "content": f"Provide feedback on the idea: '{initial_idea}'"}
                ],
                temperature=0.7,
                max_tokens=400
            )
            feedback = response.choices[0].message.content
            expert_feedback[expert_name] = feedback
            collaboration_log.append({"speaker": expert_name, "message": feedback})
            logger.debug(f"Received collaboration feedback from {expert_name}")
        except Exception as e:
            logger.error(f"Error getting collaboration feedback from {expert_name}: {e}")
            expert_feedback[expert_name] = f"Error: Could not get feedback from {expert_name}."
            collaboration_log.append({"speaker": expert_name, "message": f"Error: {e}"})

    # Super Agent synthesizes collaboration feedback
    sa_synthesis_collab_prompt = f"""
    You are the Super Agent. You have received feedback from your expert advisors on the idea:
    "{initial_idea}"

    Expert Feedback: {json.dumps(expert_feedback, indent=2)}

    Based on this feedback and your {collaboration_style} style, please:
    1.  Refine the initial idea.
    2.  Summarize the key takeaways from the collaboration.
    3.  Formulate 1-2 new questions that emerged from this collaborative discussion for further learning.

    Return ONLY a JSON object:
    {{
      "refined_idea": "...",
      "summary": "...",
      "new_questions": ["...", "..."]
    }}
    """
    try:
        sa_collab_response = call_llm_with_retry(
            super_agent_client,
            super_agent_client.models[0].id if hasattr(super_agent_client, 'models') else "gpt-3.5-turbo-0125", # Fallback for OpenAI
            messages=[{"role": "system", "content": sa_synthesis_collab_prompt}],
            temperature=0.5,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        collab_result = json.loads(sa_collab_response.choices[0].message.content)
        collaboration_log.append({"speaker": "Super Agent (Synthesis)", "message": collab_result})
        logger.debug("Super Agent collaboration synthesis complete.")
    except Exception as e:
        logger.error(f"Error during Super Agent collaboration synthesis: {e}")
        collab_result = {
            "refined_idea": f"Collaboration failed: {e}",
            "summary": "Failed to synthesize collaboration feedback.",
            "new_questions": []
        }
        collaboration_log.append({"speaker": "Super Agent (Error)", "message": f"Error: {e}"})

    return {
        "initial_idea": initial_idea,
        "expert_feedback": expert_feedback,
        "refined_idea": collab_result.get("refined_idea"),
        "summary": collab_result.get("summary"),
        "new_questions": collab_result.get("new_questions"),
        "collaboration_log": collaboration_log # Store the back-and-forth
    }
