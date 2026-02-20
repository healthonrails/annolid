---
name: Web Browsing and Automation
description: Using the embedded web viewer to fetch data, click links, and interact with pages reliably.
---

# Web Browsing and Automation

Annolid Bot has embedded web browsing capabilities accessible through `gui_web_run_steps`.
This allows you to open web pages, read their content, click elements, fill forms, and scrape data in the background.

## The Element Indexing System
When you use `gui_web_run_steps` to run a `get_text` action, the returned page text is specifically processed for AI agents.
Actionable elements (buttons, links, inputs) are tagged with unique integer numbers enclosed in brackets, e.g., `[42: button] Submit`.

### Identifying Elements
* `[15: a]` = A clickable link.
* `[22: input type=text] (placeholder: "Search")` = A text input field.
* `[3: button]` = A clickable button.

### Interacting with Elements
You do **not** need to formulate complex CSS selectors to click or type into elements. You only need to pass the **integer index** as the `selector`.

```json
{
  "steps": [
    { "action": "open_url", "url": "https://example.com/login" },
    { "action": "wait", "wait_ms": 2000 },
    { "action": "get_text", "max_chars": 5000 }
  ]
}
```

If `get_text` returns:
```text
[12: input type=text] (placeholder: "Username")
[13: input type=password] (placeholder: "Password")
[14: button] Login
```

You can log in by running another step sequence using purely the index strings as `selector`s:

```json
{
  "steps": [
    { "action": "type", "selector": "12", "text": "my_username" },
    { "action": "type", "selector": "13", "text": "my_password", "submit": true },
    { "action": "wait", "wait_ms": 3000 },
    { "action": "get_text", "max_chars": 5000 }
  ]
}
```

## Supported Actions for `gui_web_run_steps`
* `open_url` (requires `url`): Navigates the embedded browser to the target URL.
* `get_text` (optional `max_chars`): Returns the rendered text of the page, including `[index]` tags for interactive elements.
* `click` (requires `selector`): Clicks the element matching the index (or CSS selector).
* `type` (requires `selector`, `text`, optional `submit`): Sets the value of an input field and fires standard change events.
* `scroll` (requires `delta_y`): Scrolls the page vertically.
* `wait` (requires `wait_ms`): Pauses for the specified milliseconds to allow for page loads or animations.
* `find_forms`: Returns structured JSON describing all `<form>` objects and their input fields on the page.

## Best Practices
1. **Always wait for rendering**: Modern websites are dynamic. Follow an `open_url` or `click` action with a `wait` action (e.g., 2000ms - 5000ms) before trying to `get_text`.
2. **Handle truncations**: If the page is too large, `get_text` will mark `truncated` as true. In this case, use `scroll` and `get_text` in a loop to read further down.
3. **Submit via Enter key**: The `submit: true` flag in the `type` action simulates hitting the `Enter` key after typing, which is often more reliable than clicking a form's submit button.
4. **Use indices**: Never try to guess CSS selectors based on the visual text. Always use `get_text` to retrieve the numerical indices, and pass the exact integer string (e.g., `"14"`) as the `selector`.
