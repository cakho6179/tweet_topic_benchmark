"""Taxonomy, definitions, and hierarchical groupings for CardiffNLP TweetTopic 19 classes."""

from __future__ import annotations

from typing import Dict, List

TOPIC_CLASSES: List[str] = [
    "arts_&_culture",
    "business_&_entrepreneurs",
    "celebrity_&_pop_culture",
    "diaries_&_daily_life",
    "family",
    "fashion_&_style",
    "film_tv_&_video",
    "fitness_&_health",
    "food_&_dining",
    "gaming",
    "learning_&_educational",
    "music",
    "news_&_social_concern",
    "other_hobbies",
    "relationships",
    "science_&_technology",
    "sports",
    "travel_&_adventure",
    "youth_&_student_life",
]

NUM_CLASSES = len(TOPIC_CLASSES)
LABEL_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(TOPIC_CLASSES)}
ID_TO_LABEL: Dict[int, str] = {i: name for i, name in enumerate(TOPIC_CLASSES)}

# Semantic descriptions for LLM prompting & Decision Model questions
TOPIC_DESCRIPTIONS: Dict[str, str] = {
    "arts_&_culture": "Art, museums, literature, theater, poetry, architecture, and cultural events.",
    "business_&_entrepreneurs": "Finance, economics, startups, entrepreneurship, investing, stock market, and crypto.",
    "celebrity_&_pop_culture": "Famous people, viral celebrity gossip, red carpets, influencer drama, and pop trends.",
    "diaries_&_daily_life": "Personal daily experiences, morning routines, candid thoughts, and everyday life moments.",
    "family": "Parents, children, parenting, siblings, family gatherings, and home life.",
    "fashion_&_style": "Clothing, streetwear, beauty, makeup, fashion weeks, accessories, and runway trends.",
    "film_tv_&_video": "Movies, television series, Netflix, cinema, movie trailers, and YouTube videos.",
    "fitness_&_health": "Workouts, gym routines, diet, mental health, medicine, and physical wellness.",
    "food_&_dining": "Cooking, restaurants, food recipes, beverages, coffee, and delicious meals.",
    "gaming": "Video games, PC/console gaming, esports, PlayStation, Xbox, Nintendo, and streamers.",
    "learning_&_educational": "Online courses, academic study, tutorials, books, research, and skill building.",
    "music": "Songs, music albums, artists, concerts, bands, vinyl records, and musical instruments.",
    "news_&_social_concern": "Breaking news, politics, protests, world events, climate crisis, and social justice.",
    "other_hobbies": "Crafts, photography, gardening, board games, DIY projects, and personal hobbies.",
    "relationships": "Dating, marriage, friendship, romantic advice, heartbreak, and interpersonal relationships.",
    "science_&_technology": "AI, space exploration, software, gadgets, hardware, physics, and scientific discoveries.",
    "sports": "Athletics, football, basketball, tennis, leagues, tournaments, and match scores.",
    "travel_&_adventure": "Vacations, road trips, flights, tourism, hotels, scenic landscapes, and outdoor adventure.",
    "youth_&_student_life": "College, university, exams, dorm life, student humor, and teen lifestyle.",
}

# 5 Coarse Meta-Categories for Hierarchical Routing / Tree Decisions
COARSE_GROUPS: Dict[str, List[str]] = {
    "entertainment_culture": [
        "arts_&_culture",
        "celebrity_&_pop_culture",
        "film_tv_&_video",
        "gaming",
        "music",
    ],
    "personal_lifestyle": [
        "diaries_&_daily_life",
        "family",
        "fashion_&_style",
        "fitness_&_health",
        "food_&_dining",
        "relationships",
        "other_hobbies",
    ],
    "sports_travel": [
        "sports",
        "travel_&_adventure",
    ],
    "business_tech_news": [
        "business_&_entrepreneurs",
        "news_&_social_concern",
        "science_&_technology",
    ],
    "education_youth": [
        "learning_&_educational",
        "youth_&_student_life",
    ],
}
