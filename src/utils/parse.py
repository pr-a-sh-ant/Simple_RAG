import json, re

def parse_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
        d = re.search(r'[1-5]', text)
        return {"score": int(d.group()) if d else None, "reason": "unparesed"}