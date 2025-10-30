<div align="center">

# whatenc

<a href="https://pypi.org/project/whatenc/"><img src="https://img.shields.io/pypi/v/whatenc.svg" alt="PyPI"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>

Text encoding type classifier.

</div>

`whatenc` is a command-line tool that uses a gradient-boosted tree classifier to detect the encoding of a given string or file.

The model is trained on text samples from the Wikipedia corpus, with lines encoded using multiple encoding schemes to generate labeled examples.

## How It Works

`whatenc` applies a feature-based approach to characterize text, then feeds these features into a gradient-boosted decision tree model to classify the encoding.

### Feature Extraction

Each input string is converted into a feature vector describing its statistical properties.

Features include:

| Feature | Description |
| :------ | :---------- |
| Length (`n`) | Number of characters in the input |
| `n % 4` | Useful for identifying base-N encodings |
| Printable Ratio | Fraction of characters in `string.printable` |
| Alphabetic / Digit Ratios | Ratio of letters and digits to total length |
| Padding Ratio (`=`) | Common in Base64/32 encodings |
| Compressibility | Ratio of compressed to raw byte length |
| Shannon Entropy | Measure of randomness in character distribution |
| English Letter Correlation | Correlation between letter frequencies and English letter frequency distribution |
| Stopword Ratio | Fraction of English stopwords | 

### Supported Encodings

`whatenc` currently recognizes the following formats and transformations:

| Category | Encodings |
| :------- | :-------- |
| Base encodings | `base32`, `base64`, `base85`, `hex`, `url` |
| Text ciphers | `rot13`, `rot47`, `morse` |
| Compression | `gzip64` |
| Hash digests | `md5`, `sha1`, `sha224`, `sha256`, `sha384`, `sha512` |

## Installation

You can install `whatenc` using [pipx](https://pypa.github.io/pipx):

```bash
pipx install whatenc
```

## Usage

```bash
whatenc aGVsbG8gd29ybGQ=
whatenc samples.txt
```

### Examples

```bash
[+] input: aGVsbG8gd29ybGQ=
   [=] top guess   = base64
      [~] base64   = 0.455
      [~] plain    = 0.312
      [~] url      = 0.126

[+] input: hello
   [=] top guess   = plain
      [~] plain    = 0.552
      [~] url      = 0.246
      [~] rot13    = 0.192

[+] input: uryyb jbeyq
   [=] top guess   = rot13
      [~] rot13    = 0.555
      [~] plain    = 0.440
      [~] url      = 0.004
```