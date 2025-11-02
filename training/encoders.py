from functools import wraps
import base64
import binascii
import codecs
import urllib.parse
import zlib
import hashlib

def encoder(fn):
    @wraps(fn)
    def wrapper(s: str) -> str:
        try:
            return fn(s)
        except Exception:
            return ""
    return wrapper

@encoder
def base64_encode(s: str) -> str:
    return base64.b64encode(s.encode()).decode()

@encoder
def base32_encode(s: str) -> str:
    return base64.b32encode(s.encode()).decode()

@encoder
def base85_encode(s: str) -> str:
    return base64.b85encode(s.encode()).decode()

@encoder
def hex_encode(s: str) -> str:
    return binascii.hexlify(s.encode()).decode()

@encoder
def url_encode(s: str) -> str:
    return urllib.parse.quote(s)

@encoder
def rot13_encode(s: str) -> str:
    return codecs.encode(s, "rot_13")

@encoder
def rot47_encode(s: str) -> str:
    result = []
    for char in s:
        ascii_code = ord(char)
        if 33 <= ascii_code <= 126:
            result.append(chr(33 + ((ascii_code + 14) % 94)))
        else:
            result.append(char)
    return ''.join(result)

@encoder
def gzip64_encode(s: str) -> str:
    compressed = zlib.compress(s.encode())
    return base64.b64encode(compressed).decode()


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

@encoder
def morse_encode(s: str) -> str:
    s = s.upper()
    encoded = [MORSE_CODE_DICT.get(ch, '') for ch in s]
    return ' '.join(encoded)

@encoder
def md5_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

@encoder
def sha1_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

@encoder
def sha224_hash(s: str) -> str:
    return hashlib.sha224(s.encode()).hexdigest()

@encoder
def sha256_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

@encoder
def sha384_hash(s: str) -> str:
    return hashlib.sha384(s.encode()).hexdigest()

@encoder
def sha512_hash(s: str) -> str:
    return hashlib.sha512(s.encode()).hexdigest()

ENCODERS = {
    "base64": base64_encode,
    "base32": base32_encode,
    "base85": base85_encode,
    "hex": hex_encode,
    "url": url_encode,
    "rot13": rot13_encode,
    "rot47": rot47_encode,
    "gzip64": gzip64_encode,
    "morse": morse_encode,
    "md5": md5_hash,
    "sha1": sha1_hash,
    "sha224": sha224_hash,
    "sha256": sha256_hash,
    "sha384": sha384_hash,
    "sha512": sha512_hash,
}
