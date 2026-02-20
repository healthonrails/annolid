---
name: "mcp_search_optimization"
description: "Optimization guide for extracting information and using MCP browser tools effectively."
---

# MCP Browser & Search Capabilities

When you need to extract information from websites using MCP browser tools or headless evaluation scripts, avoid returning full raw HTML or massive JSON blobs back to your context.

## 1. Extract Specific Text, Not DOM
When using evaluation tools (e.g. `mcp_..._evaluate` or running scripts in the browser), do **NOT** return `document.documentElement.outerHTML` or massive node objects.

Always prefer:
- `document.body.innerText`
- `document.querySelector('main').innerText`
- Specific `.textContent` extractions

## 2. Handle Lists and Tables Carefully
If extracting many items, use a script to summarize or truncate them within the browser context *before* returning them.
Example JS execution:
```javascript
Array.from(document.querySelectorAll('.item')).slice(0, 10).map(el => el.innerText)
```

## 3. Beware Context Window Limits
If your command returns > 50,000 characters, it will be forcefully truncated by the tool wrapper to prevent you from crashing out of your iteration loop. Always aim to get exactly the text signal you need rather than downloading the entire page source.
