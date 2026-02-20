---
name: "Playwright MCP & Browser Search Optimization"
description: "Optimization guide for extracting information and using MCP browser tools effectively."
---

# MCP Browser & Search Capabilities

> [!WARNING]
> **MCP Tools are Secondary Backups for Web Browsing**
> Always prioritize native Annolid tools (`web_search`, `web_fetch`, `gui_web_run_steps`) for web interaction before falling back to Playwright or MCP tools. MCP tools should only be used if native tools fail or cannot handle highly complex JavaScript evaluations.

When you need to extract information from websites using MCP browser tools or headless evaluation scripts (like `mcp_github_com_microsoft_playwright_mcp_browser_evaluate`), it is critical to avoid returning full raw HTML or massive JSON blobs back to your context.

## 1. Extract Specific Text, Not DOM
When using evaluation tools, do **NOT** return `document.documentElement.outerHTML` or massive node objects. The Annolid MCP tool wrapper will forcefully truncate dicts/lists > 50 items and strings > 50,000 characters to prevent context overflow.

Always prefer returning clean text:
- `document.body.innerText`
- `document.querySelector('main').innerText`

## 2. Using Playwright Evaluation Scripts Properly
If you need to extract many items (e.g. search results, tables), map them to a simple Array of objects or strings *within the Javascript context* before returning.

**DO NOT DO THIS:**
```javascript
// BAD: Returns massive DOM Node references or breaks serialization
document.querySelectorAll('.product-item')
```

**DO THIS:**
```javascript
// GOOD: Returns a clean list of strings or slim JSON objects
Array.from(document.querySelectorAll('.product-item')).map(el => {
  return {
    title: el.querySelector('h2')?.innerText || '',
    price: el.querySelector('.price')?.innerText || '',
    link: el.querySelector('a')?.href || ''
  };
})
```

## 3. Playwright MCP Selectors
If you are using the Playwright MCP server's `click` or `fill` tools, you must use standard CSS selectors or Playwright text selectors.
- `"button:has-text('Submit')"`
- `'#login-form input[name="username"]'`
- `'.shopping-cart'`

## 4. Beware Context Window Limits & Truncation Warnings
If your command returns large data, `annolid` will intercept it:
1. Arrays with > 50 items will be sliced, and an annotation will be appended: `"... (X more items truncated)"`
2. Strings > 50,000 characters will be heavily truncated with a `[WARNING: MCP tool response was truncated...]` message appended.

If you see these warnings, **do not attempt to fetch the same data again unless you change your script** to target a more specific selector or paginate the results via Javascript (e.g. `.slice(50, 100)`).
