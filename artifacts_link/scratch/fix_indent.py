import re
import textwrap

with open(r'd:\Market regime detection\src\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add import textwrap
if 'import textwrap' not in content:
    content = content.replace('import json', 'import json\nimport textwrap')

# 1. render_metric_card_html
content = content.replace(
    'return f"""\n    <div class="metric-card{delay_cls}"',
    'return textwrap.dedent(f"""\n    <div class="metric-card{delay_cls}"'
)

# 2. _transition_html
content = content.replace(
    '_transition_html += f"""\n            <div class="transition-row"',
    '_transition_html += textwrap.dedent(f"""\n            <div class="transition-row"'
)

# 3. st.markdown hero-header
content = content.replace(
    'st.markdown(f"""\n    <div class="hero-header"',
    'st.markdown(textwrap.dedent(f"""\n    <div class="hero-header"'
)

# 4. render_secondary_card
content = content.replace(
    'return f"""\n            <div class="metric-card-secondary{delay_cls}"',
    'return textwrap.dedent(f"""\n            <div class="metric-card-secondary{delay_cls}"'
)

with open(r'd:\Market regime detection\src\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
