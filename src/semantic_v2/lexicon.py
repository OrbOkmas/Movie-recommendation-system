"""Small hand-written lexicons: negation cues, filler phrases, protected mood words, background-bank vocabulary.

None of these words were taken from the evaluation queries on purpose; the background bank in particular
must stay independent of M01-M28 and D01-D50 (see docs).
"""
from __future__ import annotations

NEG_CUES = {"not", "no", "without", "never", "nothing", "avoid", "avoiding", "nor", "cannot", "neither"}
COPULA = {"is", "are", "was", "were", "be", "been", "being", "am", "'s"}
AUX = {"do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "have", "has", "had"}
# after "do not" these verbs still introduce a negated *object* ("I don't want violence")
LIKE_VERBS = {"want", "like", "need", "enjoy", "love", "mind", "wish", "expect", "care", "watch", "see", "prefer"}
SPAN_BOUNDARY = {",", ";", ".", "!", "?", ":", "but", "however", "though", "although", "yet", "that", "which", "who",
                 "where", "when", "because", "so", "while", "with", "plus", "more", "just", "only", "rather", "instead",
                 "also", "then"}
SPAN_LIST_SEP = {"and", "or", "nor"}
SPAN_STOP = {"a", "an", "the", "any", "anything", "something", "too", "very", "so", "really", "much", "overly", "at",
             "all", "of", "kind", "sort", "type", "that", "is", "be", "it", "whatever", "too", "to", "be", "being",
             "extremely", "particularly", "terribly", "especially", "overtly", "either", "even", "too"}
TRAILING_JUNK = {"that", "is", "are", "was", "be", "but", "and", "which", "who", "with", "to", "a", "an", "the", "for",
                 "of", "also", "just", "only", "please", "so", "or", "then", "yet", "though", "however", "i", "we"}
LEADING_JUNK = {"but", "and", "just", "only", "also", "please", "so", "then", "yet", "though", "however", "or"}
GENERIC_NOUNS = {"film", "movie", "story", "one", "thing", "films", "movies", "stories"}

FILLER_PATTERNS = [
    r"^(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:show|give|find|get|recommend|suggest)\s+me\s+",
    r"^i(?:'d|\s+would|\s+will)?\s+(?:really\s+|just\s+|actually\s+)?(?:like|love|want|need|wish|prefer)"
    r"(?:\s+to\s+(?:watch|see|have|find|feel))?\s+",
    r"^i(?:'m|\s+am)\s+(?:in\s+the\s+mood\s+for|looking\s+for)\s+",
    r"^i\s+feel\s+like\s+(?:watching\s+)?",
    r"^(?:looking\s+for|something|anything)\s+",
    r"^(?:a|an|the|some|any|another)\s+",
    r"^(?:that|which)\s+(?:is|feels|are)\s+",
    r"\s+(?:at\s+the\s+same\s+time|please|tonight|today|right\s+now)$",
]

# Words that must never be "corrected" by the spell checker (valid but rare or informal).
PROTECTED_WORDS = {
    "tearjerker", "tearjerkers", "slapstick", "gory", "gore", "whodunit", "noir", "neo", "feelgood", "uplifting",
    "eerie", "uncanny", "melancholic", "melancholy", "brooding", "unsettling", "bittersweet", "haunting", "surreal",
    "dystopian", "cyberpunk", "rom", "com", "sci", "fi", "thriller", "thrillers", "psychedelic", "arthouse",
    "heartwarming", "gripping", "nostalgic", "dreamlike", "wistful", "claustrophobic", "cheesy", "mindless",
    "meditative", "slowburn", "coming", "ghostly", "creepy", "spooky", "quirky", "campy", "cozy", "cosy",
}

# Background-bank vocabulary for hub correction (generic mood/theme phrases; independent of any eval query).
BANK_MOODS = [
    "dark", "light-hearted", "tense", "romantic", "funny", "sad", "hopeful", "eerie", "warm", "bleak", "playful",
    "nostalgic", "gritty", "whimsical", "suspenseful", "melancholic", "thrilling", "cheerful", "gloomy", "intense",
    "peaceful", "chaotic", "tragic", "uplifting", "disturbing", "sentimental", "mysterious", "adventurous",
    "satirical", "heartbreaking", "inspiring", "violent", "gentle", "absurd", "fast-paced", "slow", "epic",
    "intimate", "creepy", "lonely", "joyful", "angry", "dreamy", "cynical", "optimistic", "spooky", "touching",
    "wild", "serious", "quirky", "scary", "exciting", "calm", "emotional", "cold", "hot-blooded", "bittersweet",
]
BANK_THEMES = [
    "family", "war", "friendship", "revenge", "love", "death", "money", "crime", "politics", "school", "travel",
    "religion", "music", "sports", "survival", "justice", "betrayal", "freedom", "memory", "ambition", "justice",
    "poverty", "immigration", "marriage", "childhood", "old age", "science", "nature", "the city", "a small town",
    "the sea", "a journey", "a secret", "a trial", "a heist", "a haunted place", "a road trip", "an island",
    "a murder", "a wedding", "a ghost", "a robbery", "a rescue", "a competition", "a search", "a disease",
]
BANK_TEMPLATES_ADJ2 = ["a {a} and {b} movie", "something {a} but {b}", "a {a} film that feels {b}"]
BANK_TEMPLATES_THEME = ["a {a} story about {t}", "a movie about {t}", "{a} drama involving {t}", "a {a} tale of {t}"]

# D8: negated word -> (genres, keyword fragments) used by the optional metadata-aware negation.
NEG_META_MAP: dict[str, dict[str, list[str]]] = {
    "scary": {"genres": ["Horror"], "keywords": ["slasher", "gore", "supernatural horror"]},
    "horror": {"genres": ["Horror"], "keywords": ["slasher", "gore"]},
    "frightening": {"genres": ["Horror"], "keywords": []},
    "creepy": {"genres": ["Horror"], "keywords": []},
    "terrifying": {"genres": ["Horror"], "keywords": []},
    "gore": {"genres": [], "keywords": ["gore", "splatter", "slasher", "blood"]},
    "gory": {"genres": [], "keywords": ["gore", "splatter", "slasher", "blood"]},
    "violence": {"genres": [], "keywords": ["violence", "gore", "murder", "massacre"]},
    "violent": {"genres": [], "keywords": ["violence", "gore", "murder", "massacre"]},
    "action": {"genres": ["Action"], "keywords": []},
    "explosions": {"genres": ["Action"], "keywords": ["explosion"]},
    "childish": {"genres": ["Family", "Animation"], "keywords": ["children", "kids"]},
    "slapstick": {"genres": [], "keywords": ["slapstick"]},
    "silly": {"genres": [], "keywords": ["slapstick", "parody"]},
    "comedy": {"genres": ["Comedy"], "keywords": []},
    "romance": {"genres": ["Romance"], "keywords": []},
    "war": {"genres": ["War"], "keywords": []},
}
