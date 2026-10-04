"""Text cleaning for the text pipeline (see PROJECT_ARCHITECTURE.md).

Order matters: emoji demojizing must happen before lowercasing (emoji package's
output uses underscores/colons that should stay as-is), and URL/mention stripping
should happen before whitespace collapsing.
"""
import re

import emoji

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet_text(text: str) -> str:
    """Clean a single raw tweet/post text string for the text pipeline."""
    if text is None:
        return ""

    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" @user ", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
