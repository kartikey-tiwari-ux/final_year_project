from src.preprocessing.text_cleaning import clean_tweet_text


def test_strips_urls():
    assert "http" not in clean_tweet_text("check this out https://example.com/foo cool")


def test_replaces_mentions_with_generic_token():
    out = clean_tweet_text("hey @john_doe how are you")
    assert "@john_doe" not in out
    assert "@user" in out


def test_demojizes_emoji_to_text():
    out = clean_tweet_text("I am so happy today 😊")
    assert "😊" not in out
    assert "smiling" in out or "happy" in out  # demojize label contains descriptive words


def test_lowercases():
    assert clean_tweet_text("HELLO World") == "hello world"


def test_collapses_whitespace():
    assert clean_tweet_text("hello    world\n\tfoo") == "hello world foo"


def test_handles_none_and_empty():
    assert clean_tweet_text(None) == ""
    assert clean_tweet_text("") == ""


def test_handles_text_with_only_url_and_mention():
    out = clean_tweet_text("https://example.com @someone")
    assert out == "@user"
