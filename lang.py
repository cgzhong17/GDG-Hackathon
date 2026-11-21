"""English / German detection: lingua restricted to {en, de} + stop-word fallback.

The dataset `language` column is never used at inference; it is only used by
evaluate.py to score this detector.
"""
import re
from functools import lru_cache

_DE = set("der die das und nicht ich bitte ist sie wir mit für fur auf ein eine einen dem den des "
          "zu von bei hallo guten tag danke sehr geehrte geehrter grüße gruesse mein meine unser "
          "unsere funktioniert kann können koennen keine wurde werden haben hat seit leider".split())
_EN = set("the and not is are you we with for on a an to of at hello hi thanks thank please dear "
          "my our it this that can cannot could does doesn't working have has since unfortunately".split())
_WORD = re.compile(r"[a-zA-ZäöüÄÖÜß']+")


@lru_cache(maxsize=1)
def _detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
        return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.GERMAN).build()
    except ImportError:
        return None


def _stopword_vote(text: str):
    words = [w.lower() for w in _WORD.findall(text)]
    de = sum(w in _DE for w in words) + 2 * len(re.findall(r"[äöüß]", text.lower()))
    en = sum(w in _EN for w in words)
    return de, en


def detect_language(subject: str, body: str, default: str = "en") -> dict:
    """Return {'lang': 'en'|'de', 'confidence': float, 'method': str}."""
    text = f"{subject or ''}\n{body or ''}".strip()
    if len(text) < 3:
        return {"lang": default, "confidence": 0.0, "method": "default (empty)"}

    det = _detector()
    if det is not None:
        conf = {c.language.name: c.value for c in det.compute_language_confidence_values(text)}
        p_de = conf.get("GERMAN", 0.0)
        if abs(p_de - 0.5) >= 0.2 or len(text) >= 40:
            lang = "de" if p_de >= 0.5 else "en"
            return {"lang": lang, "confidence": round(max(p_de, 1 - p_de), 3), "method": "lingua"}

    de, en = _stopword_vote(text)
    if de == en:
        return {"lang": default, "confidence": 0.5, "method": "default (tie)"}
    lang = "de" if de > en else "en"
    return {"lang": lang, "confidence": round(max(de, en) / (de + en), 3), "method": "stop-words"}
