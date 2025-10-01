from pathlib import Path

path = Path(r"temp_files/first_finetune.py")
text = path.read_text(encoding="utf-8")
lines = text.splitlines()
new_lines = []
prev_blank = False
for line in lines:
    if line.strip():
        new_lines.append(line)
        prev_blank = False
    else:
        if not prev_blank:
            new_lines.append("")
        prev_blank = True

for idx, line in enumerate(new_lines[:30]):
    print(idx, repr(line))
