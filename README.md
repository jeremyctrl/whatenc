# whatenc

Simple text encoding type classifier.

`whatenc` uses statistical and linguistic features to detect how a given string is encoded.

# Usage

```
pipx install whatenc
```

```
whatenc aGVsbG8gd29ybGQ=
whatenc samples.txt
```

# Examples

```
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