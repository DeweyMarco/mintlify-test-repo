# TODOs

## Streaming response support
**What:** Stream tokens as they arrive instead of waiting for the full response.
**Why:** Current UX shows nothing until the complete response is ready, which feels broken on longer answers.
**Context:** OpenAI SDK supports `stream=True`. Would need to accumulate tool calls from chunks and handle partial deltas. No blockers — purely a UX improvement for v0.2.
