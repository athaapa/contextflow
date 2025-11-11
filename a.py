import re
import json

txt = "```json\n[\n  9,\n  9,\n  7,\n  8,\n  7,\n  9,\n  7,\n  8,\n  6,\n  4,\n  8,\n  9,\n  2,\n  6,\n  2,\n  5,\n  1,\n  2,\n  1,\n  1\n]\n"
raw_text = txt.strip()

cleaned = re.sub(r"^```json\\?\n?|```$", "", raw_text, flags=re.IGNORECASE)
cleaned = cleaned.replace("\\n", "\n")  # unescape newlines

print()
