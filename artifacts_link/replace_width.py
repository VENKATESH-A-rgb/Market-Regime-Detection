import re

app_path = r"d:\Market regime detection\src\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace use_container_width=True with width="stretch"
count = content.count("use_container_width=True")
new_content = content.replace("use_container_width=True", 'width="stretch"')

with open(app_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Successfully replaced {count} occurrences of use_container_width=True with width='stretch'")
