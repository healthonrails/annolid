# Agent Box Integration

Annolid can use Box in two places:

- the agent `box` tool for list/search/download/upload workflows
- CLI helpers for OAuth bootstrap and token refresh

## Config Surface

Box settings live under `tools.box` in `~/.annolid/config.json`.

Relevant fields:

- `enabled`: enable `box` tool registration.
- `access_token`: direct access token.
- `client_id`: Box OAuth client id.
- `client_secret`: Box OAuth client secret.
- `refresh_token`: Box OAuth refresh token.
- `authorize_base_url`: Box auth host used to build the consent URL.
  - Default: `https://account.box.com`
  - Enterprise/SSO examples: `https://ent.box.com` or `https://my_org_xxx.account.box.com`
- `redirect_uri`: OAuth callback URI registered in your Box app.
  - This is the callback URL your app owns, not the Box login host.
  - The GUI suggests `http://localhost:8765/oauth/callback` by default for local use.
  - Use `http://` for local loopback callbacks. Annolid runs a plain local HTTP listener and cannot capture `https://localhost/...`.
  - Annolid reuses this across sessions once saved, so the GUI auth button does not ask again.
- `token_url`: OAuth token endpoint.
  - Default: `https://api.box.com/oauth2/token`
- `api_base`: Box API base.
  - Default: `https://api.box.com/2.0`
- `upload_api_base`: Box upload API base.
  - Default: `https://upload.box.com/api/2.0`
- `auto_refresh`: refresh access token on `401` and retry once when refresh credentials exist.

## How To Find Box App Credentials

You get the values from the Box Developer Console for the app you created for Annolid.

1. Sign in to the [Box Developer Console](https://developer.box.com/).
2. Open your app from the Platform Apps list.
3. Open the app's configuration or edit page.
4. In the OAuth 2.0 section, copy:
   - `client_id`
   - `client_secret`
5. Add the redirect URI you want Annolid to use in the app configuration page.
   - Use the same exact URI in Annolid.
   - The GUI suggests `http://localhost:8765/oauth/callback` by default for local use.
   - If you use a local callback, keep it as `http://localhost:8765/oauth/callback` or `http://127.0.0.1:8765/oauth/callback`.
6. If your Box tenant shows an org-specific login page, keep `authorize_base_url` pointed at that host instead of the generic Box host.

Practical notes:

- `client_id` is the public application identifier.
- `client_secret` is sensitive and should be treated as a secret.
- If Box asks you to verify your account before showing or copying the secret, complete that step in the developer console.
- The redirect URI must match what you registered in Box exactly.

## OAuth Bootstrap

### 1) Generate an auth URL

Use your org host if Box routes your users through an enterprise login or 2FA page:

```bash
annolid-run agent-box-auth-url \
  --client-id "$BOX_CLIENT_ID" \
  --authorize-base-url "https://my_org_xxx.account.box.com" \
  --redirect-uri "https://your-app.example.com/oauth/callback"
```

If you omit `--authorize-base-url`, Annolid uses the standard Box host.

Add `--open-browser` to launch the consent page immediately:

```bash
annolid-run agent-box-auth-url \
  --open-browser \
  --client-id "$BOX_CLIENT_ID" \
  --redirect-uri "https://your-app.example.com/oauth/callback"
```

### 2) Exchange the authorization code

```bash
annolid-run agent-box-auth-exchange \
  --client-id "$BOX_CLIENT_ID" \
  --client-secret "$BOX_CLIENT_SECRET" \
  --authorize-base-url "https://my_org_xxx.account.box.com" \
  --redirect-uri "https://your-app.example.com/oauth/callback" \
  --code "$BOX_AUTH_CODE"
```

### 3) Persist into agent config

Add `--persist` to write the tokens and Box settings into `tools.box.*`.

```bash
annolid-run agent-box-auth-exchange \
  --redirect-uri "https://your-app.example.com/oauth/callback" \
  --code "$BOX_AUTH_CODE" \
  --persist
```

### 4) Refresh an access token manually

```bash
annolid-run agent-box-token-refresh --persist
```

## Secret Hygiene

Prefer secret refs instead of plaintext config:

```bash
annolid-run agent-secrets-set --path tools.box.client_secret --env BOX_CLIENT_SECRET
annolid-run agent-secrets-set --path tools.box.refresh_token --env BOX_REFRESH_TOKEN
annolid-run agent-secrets-set --path tools.box.access_token --env BOX_ACCESS_TOKEN
```

## Runtime Behavior

- The Box tool refreshes automatically when `auto_refresh=true` and the config includes `client_id`, `client_secret`, and `refresh_token`.
- If Box returns `401`, Annolid retries once after refreshing the access token.
- CLI auth helpers read missing fields from `tools.box.*`, so you can keep repetitive values in config and only override what changes.
- The Agent Runtime settings dialog includes editable Box auth fields for `authorize_base_url`, `client_id`, `client_secret`, and `redirect_uri`.
- The redirect field includes a copy button so you can paste the callback URL into the Box app registration page.
- The auth button now runs the full local browser OAuth flow when the redirect URI is loopback.
- Annolid starts a temporary callback listener on `http://localhost` or `http://127.0.0.1`, opens the Box consent page, captures the redirect, exchanges the code, and saves the tokens.
- If you still see `localhost refused to connect`, the redirect URI is not matching the registered Box callback exactly, or you used `https://localhost/...` instead of `http://localhost/...`.
- The button works across Annolid sessions because the saved config retains the Box host, client ID, client secret, redirect URI, access token, and refresh token.

## What To Ask The Agent

Use natural language tasks. A few examples:

- "List the files in my Box project folder and summarize the latest uploads."
- "Find the Box folder that contains the March motion capture exports."
- "Download the newest CSV from Box into my Annolid workspace."
- "Upload the results in `results/motion.csv` to the Box project folder."
- "Check whether the Box folder already has a tracking export for this session."
- "Search Box for a file named `session_042_annotations.json` and give me the file id."
- "Replace the existing Box file version with the updated CSV."
- "Compare the Box folder contents against the local analysis outputs and tell me what is missing."

Useful patterns:

- Ask for `list`, `search`, `get file info`, `download`, or `upload` by describing the goal, not the API name.
- Include the Box folder name, file name, or session label if you know it.
- Tell the agent whether to overwrite an existing file or create a new version.
- If the Box folder is nested in an enterprise tenant, the saved `authorize_base_url` keeps the auth flow pointed at your organization host.

## Practical Tips

- For enterprise tenants, set `authorize_base_url` to the host Box uses for your organization, such as `https://ent.box.com` or `https://my_org_xxx.account.box.com`.
- If you do not see your app's credentials immediately, use the app details or configuration page in the Box Developer Console. That is where Box shows the Client ID and secret.
- If auth succeeds but tool calls still fail, verify that the refresh token is still valid and that the Box app scopes include the operations you want.
- If the GUI says `Missing Box client secret`, fill in the secret from the Box Developer Console or use the secret-ref workflow in `docs/agent_secrets.md`.
- If the local callback page refuses to connect, check that the redirect URI registered in Box matches Annolid exactly and uses `http://` for `localhost` or `127.0.0.1`.
