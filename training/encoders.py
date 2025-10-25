import base64
import binascii
import codecs
import urllib.parse
import zlib


def base64_encode(s: str) -> str:
    try:
        return base64.b64encode(s.encode()).decode()
    except Exception:
        return ""


def base32_encode(s: str) -> str:
    try:
        return base64.b32encode(s.encode()).decode()
    except Exception:
        return ""


def base85_encode(s: str) -> str:
    try:
        return base64.b85encode(s.encode()).decode()
    except Exception:
        return ""


def hex_encode(s: str) -> str:
    try:
        return binascii.hexlify(s.encode()).decode()
    except Exception:
        return ""


def url_encode(s: str) -> str:
    try:
        return urllib.parse.quote(s)
    except Exception:
        return ""


def rot13_encode(s: str) -> str:
    try:
        return codecs.encode(s, "rot_13")
    except Exception:
        return ""


def gzip64_encode(s: str) -> str:
    try:
        compressed = zlib.compress(s.encode())
        return base64.b64encode(compressed).decode()
    except Exception:
        return ""


MORSE_CODE_DICT = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..",
    "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-",
    "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..",
    "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..",
    "9": "----.", "0": "-----", ", ": "--..--", ".": ".-.-.-",
    "?": "..--..", "/": "-..-.", "-": "-....-", "(": "-.--.", ")": "-.--.-",
}


def morse_encode(s: str) -> str:
    s = s.upper()
    encoded = [MORSE_CODE_DICT.get(ch, '') for ch in s]
    return ' '.join(encoded)

ENCODERS = {
    "base64": base64_encode,
    "base32": base32_encode,
    "base85": base85_encode,
    "hex": hex_encode,
    "url": url_encode,
    "rot13": rot13_encode,
    "gzip64": gzip64_encode,
    "morse": morse_encode,
}
