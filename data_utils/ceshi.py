source_line = "<s> vs. electric : is hydrogen superior to electric vehicles ? <eos>"
a = " ".join(source_line.split()[1:-1]).strip()
if a.find(":"):
    pos = a.find(":")
    a = a[pos + 1:].strip()
print(a)
print(pos)