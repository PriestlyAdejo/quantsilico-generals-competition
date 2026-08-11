from pathlib import Path

for rel in (
    "scripts/wsl/_emergency_phase_a_and_launch_ppo.sh",
):
    p = Path(rel)
    data = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    p.write_bytes(data)
    print("fixed", rel, len(data))
