# _agent_profiles.py

SUPER_AGENT_PROFILES = {
    "technical_master": {
        "profile_name": "Technical Master Super Agent",
        "learning_style": "analytical, detail-oriented, empirical",
        "dreaming_tendency": "medium", # Controls how often/extensively it 'dreams'
        "current_knowledge_state": "novice", # This will be updated dynamically
        "collaboration_style": "structured critique and problem-solving",
        "focus_areas": ["precision", "accuracy", "depth in specific domains"]
    },
    "creative_innovator": {
        "profile_name": "Creative Innovator Super Agent",
        "learning_style": "exploratory, associative, hypothetical",
        "dreaming_tendency": "high",
        "current_knowledge_state": "novice",
        "collaboration_style": "free-association brainstorming and ideation",
        "focus_areas": ["novel applications", "interdisciplinary connections", "future implications"]
    },
    "critical_philosopher": {
        "profile_name": "Critical Philosopher Super Agent",
        "learning_style": "skeptical, ethical, reflective",
        "dreaming_tendency": "low",
        "current_knowledge_state": "novice",
        "collaboration_style": "devil's advocate and ethical review",
        "focus_areas": ["biases", "limitations", "societal impact", "ethical considerations"]
    }
}

EXPERT_AGENT_PROFILES = {
    "openai_gpt": { # This would be your internal GPT-3.5 or GPT-4 for expert roles
        "profile_name": "OpenAI GPT Expert",
        "role": "general knowledge, foundational concepts, clear explanations",
        "model": "gpt-3.5-turbo-0125", # Or gpt-4o, etc.
        "collaboration_mode": "structured feedback and summary"
    },
    "google_gemini": {
        "profile_name": "Google Gemini Expert",
        "role": "up-to-date information, creative writing, multi-modal perspectives",
        "model": "gemini-1.5-flash", # Or gemini-1.5-pro
        "collaboration_mode": "diverse perspective generation and analogy creation"
    },
    "grok_expert": {
        "profile_name": "Grok Expert",
        "role": "unconventional insights, humorous angles, current event connections",
        "model": "grok-1", # If you have API access
        "collaboration_mode": "outside-the-box brainstorming and challenging assumptions"
    },
    "claude_expert": {
        "profile_name": "Anthropic Claude Expert",
        "role": "safety-focused, harmless and helpful, long-context understanding",
        "model": "claude-3-opus-20240229", # Or other Claude models
        "collaboration_mode": "safety review, ethical considerations, and nuanced analysis"
    },
    "mistral_expert": {
        "profile_name": "Mistral Expert",
        "role": "efficient reasoning, code generation, multilingual capabilities",
        "model": "mistral-large-latest", # Or other Mistral models
        "collaboration_mode": "technical solution proposing and efficiency optimization"
    }
    # Add your "2 other LLMs" here with their profiles
}
