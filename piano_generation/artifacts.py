import re

placeholder_tokens = [f"<SENTINEL_{idx}>" for idx in range(64)]
dataset_tokens = [
    "<MAESTRO>",
    "<PIJAMA>",
    "<VGMIDI>",
    "<MUSIC-NET>",
    "<PIANO-MIDI-DE>",
    "<LAKH-LMD-FULL>",
    "<GIANT-MIDI>",
    "<IMSLP>",
    "<ATEPP-1.1>",
    "<PIANO_FOR_AI>",
]
composer_tokens = [
    "<SCRIABIN>",
    "<FRANCK>",
    "<MOZART>",
    "<CHOPIN>",
    "<MENDELSSON>",
    "<LISZT>",
    "<SCHUBERT>",
    "<BRAHMS>",
    "<HAYDN>",
    "<BEETHOVEN>",
    "<BALAKIREV>",
    "<SCHUMANN>",
    "<RACHMANIOFF>",
    "<UNKNOWN_COMPOSER>",
    "<BACH>",
]
special_tokens = (
    [
        "<PAD>",
        "<CLS>",
        "<EOS>",
        "<SENTINEL_78>",
        "<SENTINEL_79>",
        "<SENTINEL_80>",
        "<SENTINEL_81>",
        "<SENTINEL_82>",
        "<SENTINEL_83>",
        "<SENTINEL_84>",
        "<SENTINEL_85>",
        "<SENTINEL_86>",
        "<SENTINEL_87>",
        "<SENTINEL_88>",
        "<SENTINEL_89>",
        "<SENTINEL_90>",
        "<SENTINEL_91>",
        "<SENTINEL_92>",
        "<SENTINEL_93>",
        "<SENTINEL_94>",
        "<SENTINEL_95>",
        "<SCORES>",
        "<PERFORMANCE>",
        "<CLEAN_TIME>",
        "<CLEAN_EVERYTHING>",
        "<CLEAN_VOLUME>",
        "<CLEAN_PITCH>",
        "<LOW_FROM_MEDIAN>",
        "<HIGH_FROM_MEDIAN>",
        "<ABOVE_LOW_QUARTILE>",
        "<BELOW_LOW_QUARTILE>",
        "<ABOVE_HIGH_QUARTILE>",
        "<BELOW_HIGH_QUARTILE>",
        "<MIDDLE_QUARTILES>",
        "<EXTREME_QUARTILES>",
        "<LOUD>",
        "<SOFT>",
        "<ABOVE_VERY_SOFT>",
        "<VERY_SOFT>",
        "<VERY_LOUD>",
        "<BELOW_VERY_LOUD>",
        "<MODERATE_VOLUME>",
        "<EXTREME_VOLUME>",
        "<CLEAN>",
        "<NOISY_VOLUME>",
        "<NOISY_PITCH>",
        "<NOISY_START_TIME>",
        "<NOISY_TIME>",
        "<NOISY>",
    ]
    + composer_tokens
    + placeholder_tokens
)

composer_token_map: dict[str, str] = {
    "Alexander Scriabin": "<SCRIABIN>",
    "César Franck": "<FRANCK>",
    "Wolfgang Amadeus Mozart": "<MOZART>",
    "Frédéric Chopin": "<CHOPIN>",
    "Felix Mendelssohn": "<MENDELSSON>",
    "Franz Liszt": "<LISZT>",
    "Franz Schubert": "<SCHUBERT>",
    "Johannes Brahms": "<BRAHMS>",
    "Joseph Haydn": "<HAYDN>",
    "Ludwig van Beethoven": "<BEETHOVEN>",
    "Mily Balakirev": "<BALAKIREV>",
    "Robert Schumann": "<SCHUMANN>",
    "Sergei Rachmaninoff": "<RACHMANIOFF>",
    "Johann Sebastian Bach": "<BACH>",
}


def create_composer_regex_map() -> dict[re.Pattern, str]:
    regex_map: dict[re.Pattern, str] = {}
    for full_name, token in composer_token_map.items():
        names = full_name.split()
        surname = names[-1]
        pattern = re.compile(rf"\b{re.escape(surname)}\b", re.IGNORECASE)
        regex_map[pattern] = token
    return regex_map


composer_regex_map: dict[re.Pattern, str] = create_composer_regex_map()


def get_composer_token(composer: str) -> str:
    matches: list[tuple[re.Match, str]] = [
        (match, token) for pattern, token in composer_regex_map.items() if (match := pattern.search(composer))
    ]

    if len(matches) == 1:
        return matches[0][1]
    return "<UNKNOWN_COMPOSER>"
