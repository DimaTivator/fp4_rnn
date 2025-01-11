utf8_codes = list(map(int, input().split()))
chars = ''.join(chr(code) for code in utf8_codes)
print("Answer:", chars)
