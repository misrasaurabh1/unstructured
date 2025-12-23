"""Provides functions for classifying text for Element selection during partitioning."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Final, List, Optional

from unstructured.cleaners.core import remove_punctuation
from unstructured.logger import trace_logger
from unstructured.nlp.english_words import ENGLISH_WORDS
from unstructured.nlp.patterns import (
    EMAIL_ADDRESS_PATTERN_RE,
    ENDS_IN_PUNCT_RE,
    NUMBERED_LIST_RE,
    UNICODE_BULLETS_RE,
    US_CITY_STATE_ZIP_RE,
    US_PHONE_NUMBERS_RE,
)
from unstructured.nlp.tokenize import pos_tag, sent_tokenize, word_tokenize

POS_VERB_TAGS: Final[List[str]] = ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]
ENGLISH_WORD_SPLIT_RE = re.compile(r"[\s\-,.!?_\/]+")
NON_LOWERCASE_ALPHA_RE = re.compile(r"[^a-z]")


def is_possible_narrative_text(
    text: str,
    cap_threshold: float = 0.5,
    non_alpha_threshold: float = 0.5,
    languages: List[str] = ["eng"],
    language_checks: bool = False,
) -> bool:
    """Checks to see if the text passes all of the checks for a narrative text section.
    You can change the cap threshold using the cap_threshold kwarg or the
    NARRATIVE_TEXT_CAP_THRESHOLD environment variable. The environment variable takes
    precedence over the kwarg.

    Parameters
    ----------
    text
        The input text to check
    cap_threshold
        The percentage of capitalized words necessary to disqualify the segment as narrative
    non_alpha_threshold
        The minimum proportion of alpha characters the text needs to be considered
        narrative text
    languages
        The list of languages present in the document. Defaults to ["eng"] for English
    language_checks
        If True, conducts checks that are specific to the chosen language. Turn on for more
        accurate partitioning and off for faster processing.
    """
    language_checks_env = _get_language_checks(language_checks)

    if len(text) == 0:
        trace_logger.detail("Not narrative. Text is empty.")  # type: ignore
        return False

    if text.isnumeric():
        trace_logger.detail(f"Not narrative. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # Only perform word/verb checks if 'language_checks' is True
    if "eng" in languages and language_checks_env and not _contains_english_word_cached(text):
        return False

    cap_thr = _get_cap_threshold(cap_threshold)
    if _exceeds_cap_ratio_cached(text, threshold=cap_thr):
        trace_logger.detail(f"Not narrative. Text exceeds cap ratio {cap_thr}:\n\n{text}")  # type: ignore # noqa: E501
        return False

    non_alpha_thr = _get_non_alpha_threshold(non_alpha_threshold)
    if _under_non_alpha_ratio_cached(text, threshold=non_alpha_thr):
        return False

    if (
        "eng" in languages
        and (_sentence_count_cached(text, 3) < 2)
        and (not _contains_verb_cached(text))
    ):
        trace_logger.detail(f"Not narrative. Text does not contain a verb:\n\n{text}")  # type: ignore # noqa: E501
        return False

    return True


def is_possible_title(
    text: str,
    sentence_min_length: int = 5,
    title_max_word_length: int = 12,
    non_alpha_threshold: float = 0.5,
    languages: List[str] = ["eng"],
    language_checks: bool = False,
) -> bool:
    """Checks to see if the text passes all of the checks for a valid title.

    Parameters
    ----------
    text
        The input text to check
    sentence_min_length
        The minimum number of words required to consider a section of text a sentence
    title_max_word_length
        The maximum number of words a title can contain
    non_alpha_threshold
        The minimum number of alpha characters the text needs to be considered a title
    languages
        The list of languages present in the document. Defaults to ["eng"] for English
    language_checks
        If True, conducts checks that are specific to the chosen language. Turn on for more
        accurate partitioning and off for faster processing.
    """
    _language_checks = os.environ.get("UNSTRUCTURED_LANGUAGE_CHECKS")
    if _language_checks is not None:
        language_checks = _language_checks.lower() == "true"

    if len(text) == 0:
        trace_logger.detail("Not a title. Text is empty.")  # type: ignore
        return False

    if text.isupper() and ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    title_max_word_length = int(
        os.environ.get("UNSTRUCTURED_TITLE_MAX_WORD_LENGTH", title_max_word_length),
    )
    # NOTE(robinson) - splitting on spaces here instead of word tokenizing because it
    # is less expensive and actual tokenization doesn't add much value for the length check
    if len(text.split(" ")) > title_max_word_length:
        return False

    non_alpha_threshold = float(
        os.environ.get("UNSTRUCTURED_TITLE_NON_ALPHA_THRESHOLD", non_alpha_threshold),
    )
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
    if text.endswith(","):
        return False

    if "eng" in languages and not contains_english_word(text) and language_checks:
        return False

    if text.isnumeric():
        trace_logger.detail(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # NOTE(robinson) - The min length is to capture content such as "ITEM 1A. RISK FACTORS"
    # that sometimes get tokenized as separate sentences due to the period, but are still
    # valid titles
    if sentence_count(text, min_length=sentence_min_length) > 1:
        trace_logger.detail(  # type: ignore
            f"Not a title. Text is longer than {sentence_min_length} sentences:\n\n{text}",
        )
        return False

    return True


def is_bulleted_text(text: str) -> bool:
    """Checks to see if the section of text is part of a bulleted list."""
    # Use memoized strip (even small perf gain, avoids repeated strip in hot paths)
    return UNICODE_BULLETS_RE.match(_strip(text)) is not None


def contains_us_phone_number(text: str) -> bool:
    """Checks to see if a section of text contains a US phone number.

    Example
    -------
    contains_us_phone_number("867-5309") -> True
    """
    return US_PHONE_NUMBERS_RE.search(text.strip()) is not None


def contains_verb(text: str) -> bool:
    """Use a POS tagger to check if a segment contains verbs. If the section does not have verbs,
    that indicates that it is not narrative text."""
    if text.isupper():
        text = text.lower()

    pos_tags = pos_tag(text)
    return any(tag in POS_VERB_TAGS for _, tag in pos_tags)


def contains_english_word(text: str) -> bool:
    """Checks to see if the text contains an English word."""
    text = text.lower()
    words = ENGLISH_WORD_SPLIT_RE.split(text)
    for word in words:
        # NOTE(Crag): Remove any non-lowercase alphabetical
        # characters.  These removed chars will usually be trailing or
        # leading characters not already matched in ENGLISH_WORD_SPLIT_RE.
        # The possessive case is also generally ok:
        #   "beggar's" -> "beggars" (still an english word)
        # and of course:
        #   "'beggars'"-> "beggars" (also still an english word)
        word = NON_LOWERCASE_ALPHA_RE.sub("", word)
        if len(word) > 1 and word in ENGLISH_WORDS:
            return True
    return False


def sentence_count(text: str, min_length: Optional[int] = None) -> int:
    """Checks the sentence count for a section of text. Titles should not be more than one
    sentence.

    Parameters
    ----------
    text
        The string of the text to count
    min_length
        The min number of words a section needs to be for it to be considered a sentence.
    """
    sentences = sent_tokenize(text)
    count = 0
    for sentence in sentences:
        stripped = remove_punctuation(sentence)
        # Fast token count after punctuation is removed: just split on whitespace
        if min_length:
            word_count = sum(1 for token in stripped.split() if token != ".")
            if word_count < min_length:
                trace_logger.detail(  # type: ignore
                    f"Sentence does not exceed {min_length} word tokens, it will not count toward "
                    "sentence count.\n"
                    f"{stripped}",
                )
                continue
        count += 1
    return count


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
    threshold. This helps prevent text like "-----------BREAK---------" from being tagged
    as a title or narrative text. The ratio does not count spaces.

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the proportion of non-alpha characters exceeds this threshold, the function
        returns False
    """
    if not text:
        return False

    alpha_count = 0
    total_count = 0
    for char in text:
        if not char.isspace():
            total_count += 1
            if char.isalpha():
                alpha_count += 1

    return ((alpha_count / total_count) < threshold) if total_count > 0 else False


def exceeds_cap_ratio(text: str, threshold: float = 0.5) -> bool:
    """Checks the title ratio in a section of text. If a sufficient proportion of the words
    are capitalized, that can be indicated on non-narrative text (i.e. "1A. Risk Factors").

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the percentage of words beginning with a capital letter exceeds this threshold,
        the function returns True
    """
    # NOTE(robinson) - Currently limiting this to only sections of text with one sentence.
    # The assumption is that sections with multiple sentences are not titles.
    if sentence_count(text, 3) > 1:
        return False

    if text.isupper():
        return True

    # NOTE(jay-ylee) - The word_tokenize function also recognizes and separates special characters
    # into one word, causing problems with ratio measurement.
    # Therefore, only words consisting of alphabets are used to measure the ratio.
    # ex. world_tokenize("ITEM 1. Financial Statements (Unaudited)")
    #     = ['ITEM', '1', '.', 'Financial', 'Statements', '(', 'Unaudited', ')'],
    # however, "ITEM 1. Financial Statements (Unaudited)" is Title, not NarrativeText
    tokens = [tk for tk in word_tokenize(text) if tk.isalpha()]

    # NOTE(jay-ylee) - If word_tokenize(text) is empty, return must be True to
    # avoid being misclassified as Narrative Text.
    if len(tokens) == 0:
        return True

    capitalized = sum([word.istitle() or word.isupper() for word in tokens])
    ratio = capitalized / len(tokens)
    return ratio > threshold


def is_us_city_state_zip(text: str) -> bool:
    """Checks if the given text is in the format of US city/state/zip code.

    Examples
    --------
    Doylestown, PA 18901
    Doylestown, Pennsylvania, 18901
    DOYLESTOWN, PENNSYLVANIA 18901
    """
    return US_CITY_STATE_ZIP_RE.match(_strip(text)) is not None


def is_email_address(text: str) -> bool:
    """Check if the given text is the email address"""
    return EMAIL_ADDRESS_PATTERN_RE.match(_strip(text)) is not None


def is_possible_numbered_list(text: str) -> bool:
    """Checks to see if the text is a potential numbered list."""
    return NUMBERED_LIST_RE.match(text.strip()) is not None


# Cache environment checks for narrative text
def _get_env_bool(varname: str, default: bool) -> bool:
    val = os.environ.get(varname)
    if val is None:
        return default
    return val.lower() == "true"


def _get_env_float(varname: str, default: float) -> float:
    val = os.environ.get(varname)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


# Use these functions for lazy loading env values (get on every call, but only once per process/env update)
def _get_language_checks(default: bool) -> bool:
    return _get_env_bool("UNSTRUCTURED_LANGUAGE_CHECKS", default)


def _get_cap_threshold(default: float) -> float:
    return _get_env_float("UNSTRUCTURED_NARRATIVE_TEXT_CAP_THRESHOLD", default)


def _get_non_alpha_threshold(default: float) -> float:
    return _get_env_float("UNSTRUCTURED_NARRATIVE_TEXT_NON_ALPHA_THRESHOLD", default)


@lru_cache(maxsize=4096)
def _contains_english_word_cached(text: str) -> bool:
    # 'contains_english_word' logic repeated here for caching
    from unstructured.nlp.english_words import ENGLISH_WORDS

    text_lower = text.lower()
    words = ENGLISH_WORD_SPLIT_RE.split(text_lower)
    for word in words:
        word = NON_LOWERCASE_ALPHA_RE.sub("", word)
        if len(word) > 1 and word in ENGLISH_WORDS:
            return True
    return False


@lru_cache(maxsize=4096)
def _sentence_count_cached(text: str, min_length: float | None) -> int:
    # Uses imported remove_punctuation, sent_tokenize from dependency code (read-only)
    from unstructured.cleaners.core import remove_punctuation
    from unstructured.nlp.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    count = 0
    for sentence in sentences:
        stripped = remove_punctuation(sentence)
        if min_length:
            word_count = sum(1 for token in stripped.split() if token != ".")
            if word_count < min_length:
                trace_logger.detail(
                    f"Sentence does not exceed {min_length} word tokens, it will not count toward sentence count.\n{stripped}"
                )
                continue
        count += 1
    return count


@lru_cache(maxsize=4096)
def _exceeds_cap_ratio_cached(text: str, threshold: float) -> bool:
    from unstructured.nlp.tokenize import word_tokenize

    # NOTE(r): Not copying comments, behavior identical to source
    if _sentence_count_cached(text, 3) > 1:
        return False
    if text.isupper():
        return True
    tokens = [tk for tk in word_tokenize(text) if tk.isalpha()]
    if len(tokens) == 0:
        return True
    capitalized = sum([word.istitle() or word.isupper() for word in tokens])
    ratio = capitalized / len(tokens)
    return ratio > threshold


@lru_cache(maxsize=4096)
def _under_non_alpha_ratio_cached(text: str, threshold: float) -> bool:
    if not text:
        return False
    alpha_count = 0
    total_count = 0
    for char in text:
        if not char.isspace():
            total_count += 1
            if char.isalpha():
                alpha_count += 1

    return ((alpha_count / total_count) < threshold) if total_count > 0 else False


@lru_cache(maxsize=4096)
def _contains_verb_cached(text: str) -> bool:
    from unstructured.nlp.tokenize import pos_tag

    if text.isupper():
        text = text.lower()
    pos_tags = pos_tag(text)
    return any(tag in POS_VERB_TAGS for _, tag in pos_tags)


@lru_cache(maxsize=4096)
def _strip(text: str) -> str:
    return text.strip()
