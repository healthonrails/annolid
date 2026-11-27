---
name: cloud-workspace
description: Work with Cornell Box, Google Drive, Google Calendar, and Email tools to browse, upload, download, schedule, and communicate cleanly without guessing local mount paths.
metadata: '{"annolid":{"always":false,"tags":["box","cornell box","google drive","drive","gdrive","google calendar","calendar","email","inbox","upload","download","cloud storage","meeting","event"]}}'
---

# Cloud Storage, Calendar & Email

Use this skill when the user asks to browse, search, upload, or download files on Cornell Box or Google Drive, manage Google Calendar events, or read/send emails.

## 1. Cornell Box (`box`)

Always use the `box` tool for Box/Cornell Box requests — **never** guess local mount paths with `list_dir`.

- `box(action="list_folder_items", folder_id="0", limit?, offset?)`
- `box(action="search", query="...", item_type?, limit?)`
- `box(action="get_file_info", file_id="...")`
- `box(action="download_file", file_id="...", destination_path="...", overwrite?)`
- `box(action="upload_file", file_path="...", folder_id="...", file_name?)`

## 2. Google Drive (`google_drive`)

- Browse/search: `google_drive(action="list_files", query?, parent_id?, max_results?, supports_all_drives?)`
- Inspect metadata: `google_drive(action="get_file", file_id="...")`
- Organize: `google_drive(action="create_folder", folder_name="...", parent_id?)`
- Transfer files:
  - `google_drive(action="upload_file", local_path="...", remote_folder_id?, remote_folder_path?, skip_if_exists?)`
  - `google_drive(action="download_file", file_id="...", local_path="...")`
- Batch video backup:
  - `google_drive(action="upload_saved_videos"|"upload_realtime_videos", source_dir?, remote_folder_path?, max_files?, modified_within_hours?, skip_if_exists?)`
- Delete (only with explicit confirmation): `google_drive(action="delete_file", file_id="...")`

## 3. Google Calendar (`google_calendar`)

- List upcoming events: `google_calendar(action="list_events", start_time?, end_time?, max_results?)`
- Create/update/delete:
  - `google_calendar(action="create_event", summary="...", start_time="...", end_time="...", description?, location?, recurrence_rule?)`
  - `google_calendar(action="update_event", event_id="...", ...)`
  - `google_calendar(action="delete_event", event_id="...")`
- Use ISO-8601 timestamps with explicit timezone offsets and confirm before deleting or rescheduling existing meetings.

## 4. Email (`list_emails`, `read_email`, `email`)

- Read inbox: `list_emails(count?, unread_only?)` -> `read_email(email_id="...")`
- Send email: `email(to="...", subject="...", content="...", attachment_paths?)`
- For delayed or recurring emails, use `cron` (see the `cron` skill).

## Safety & Guardrails

1. Confirm recipients, attachment paths, and file deletions before mutating external state.
2. Keep local upload/download paths inside allowed workspace roots and report exact file IDs, remote paths, or event IDs on completion.
3. If an auth token or service credential is missing, report the exact tool error and setup step rather than inventing fallback workarounds.
