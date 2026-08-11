from pathlib import Path

for p in Path("scripts/wsl").glob("_emergency*.sh"):
    data = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    p.write_bytes(data)
    print("fixed", p)
