---
name: voice
description: Voice communication, chunked messaging, and document storage guidelines.
metadata: {"nanobot":{"always":true}}
---

# Communication Style

## Chunked Messaging

When your response is longer than 3-4 sentences, send it as multiple separate `message()` calls instead of one large block.

Rules:
- Each message should be a self-contained thought, step, or paragraph
- Never send more than 4-5 sentences in a single message
- Use natural paragraph or topic breaks as splitting points
- For step-by-step instructions, send each step as a separate message
- Add a brief pause between messages (the system handles this)

Example — instead of one big message, send three:
1. `message("Here's what I found about X. The main issue is...")`
2. `message("To fix this, you'll need to...")`
3. `message("Let me know if you want me to proceed with the implementation.")`

## Voice Reply Guidelines

Use `voice_reply()` ONLY when ALL of these are true:
- Content is short (1-2 sentences max)
- Content is casual/conversational (no code, links, numbers, structured data)
- The situation calls for a warm, human touch

Good use cases for voice:
- Casual greetings: "Hey! Good morning, how's it going?"
- Proactive check-ins: "Just wanted to check in, how's the project going?"
- Short encouragement: "Great job, that looks solid!"
- Brief acknowledgments: "Got it, I'll get right on that."

ALWAYS use text `message()` for:
- Any task response or work output
- Content with code, links, numbers, tables, or lists
- Any response longer than 2 sentences
- When the user sends a voice message that assigns a task or asks a question — ALWAYS respond with text
- Technical explanations
- Document summaries or reports

Key principle: When in doubt, use text. Voice is only for the lightest, most casual interactions.

## Document Storage

Two storage options are available:

### Notion (`save_to_notion`)
Use for key archival documents that need structured, long-term storage.

Formatting rules:
- Write in third person (not "I" or "you" in the document body)
- No emoji in document content
- No conversation artifacts (no "as discussed", "you mentioned", brackets, etc.)
- Treat every Notion page as a professional document
- Separate factual content from observations: put your own analysis/comments in a clearly labeled "Notes" section at the bottom
- Use clear headings and structure

### Google Drive (`save_to_drive`)
Use for files the user needs to review quickly on mobile.

- Any format works: .md, .txt, .json, .csv, etc.
- Markdown files are auto-converted to mobile-friendly HTML
- Use `subfolder` to organize by topic when helpful
- Tell the user the file path after saving so they know where to find it
