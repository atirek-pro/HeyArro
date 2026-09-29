# Hey Arro

Hey Arro is a Windows desktop AI teaching companion that sits next to your cursor.

Hold a hotkey, ask a question out loud, and Hey Arro looks at your screen, works out
what you are asking about, and then **teaches** you: it moves its own pointer to the
thing on screen, draws a highlight/box/circle/underline around it, and explains it in
spoken steps - each sentence appearing only after the matching visual is up, and the
next step only starting once that sentence has finished.

It is a local prototype, built to be read: a small provider-based Python application
where the microphone, the transcription engine, the vision model, the voice and the
overlay are all swappable behind narrow interfaces.

```
Ctrl + Alt held          ->  record the microphone
Ctrl + Alt released      ->  transcribe locally + capture the screen
                         ->  ask the vision model, validated into a typed response
                         ->  turn the answer into a teaching plan (with a depth level)
                         ->  ground every visual target  <-- preparation ends here
                         ->  teach: show visuals -> speak -> wait -> next step
                         ->  idle, waiting for your next question
```

---

## Features

**Push-to-talk, anywhere in Windows**
Hold `Ctrl + Alt` while Hey Arro is unfocused, speak, release. Pressing again
interrupts whatever it is still saying and starts over.

**Your voice stays on your machine**
Speech is transcribed locally with faster-whisper (CPU `int8` by default, model size
configurable, language auto-detected). Audio is never uploaded.

**It answers about what you are looking at**
One screenshot per question (monitor selectable). The model is told the screenshot's
exact pixel grid, so the coordinates it returns actually mean something.

**Answers are structured data, not prose**
Every response is validated into a typed object - tone, teaching mode, ordered steps,
and each step's visual actions. A malformed answer becomes a clear error instead of a
plausible-looking one, and no vendor SDK type ever leaves its provider module.

**Voice and visuals taught in step**
For each step the visuals appear first, then the sentence is spoken, and the next step
waits for that sentence to end. Nothing is ever spoken twice.

**Depth follows what you asked for**
`beginner`, `intermediate` and `advanced` are decided from *your wording* ("explain
this like I'm a beginner", "give me the technical implementation details"), never from
guesses about you - a technically hard question does not make you an advanced learner.
The level changes the language, the amount of context and the size of the steps; it
never changes what is being taught.

**You can keep the conversation going**
"Explain that again", "make it simpler", "go deeper", "show me another example",
"I don't understand step 2", "what should I remember?", "continue". Each one plans a
*new* lesson that continues from the last, grounded against the screen as it is now.
A recap or a repeat is answered in words, without touching the screen. Nothing here is
automatic: Hey Arro only continues when you ask it to.

**Pointing that lands on the right thing**
The first pass's targets are approximate, so each one is refined with a close-up,
high-resolution second pass; a low-confidence refinement is rejected and the original
target is kept. A lesson therefore degrades gracefully instead of breaking - and it
never invents a target it cannot see.

**A pointer, not a screenshot**
A transparent, click-through, always-on-top overlay draws Arro's own animated pointer
plus `point`, `highlight`, `box`, `circle` and `underline` primitives over your real
screen. Nothing is clicked and nothing is typed: Hey Arro explains, it never operates
your machine.

**All preparation before any teaching**
Every visual target is grounded before the first word is spoken, so there are no model
calls, no screenshot captures and no pauses *between* steps. You wait once, at the
start, and then it just teaches.

**Runs without any keys at all**
`LLM_PROVIDER=mock` gives an offline echo response, `TTS_PROVIDER=mock` keeps it
silent, and `TEACHING_ENABLED=false` runs it voice-only - so the whole pipeline can be
exercised with no API key, no microphone and no display overlay.

**Everything is tested**
Over 500 tests cover the state machine, the coordination, the response contract, the
plan model and conversion, difficulty, follow-ups, grounding, speech ordering,
cancellation and the overlay. None of them needs an API key, a microphone or a real
model.

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (the global hotkey and the built-in voice are Windows-specific) |
| Python | 3.12 (developed on 3.12.10) |
| Hardware | a microphone, and a normal desktop with a display |
| Network | needed once to download the Whisper model, and then for the vision model |
| Key | a personal Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) - free tier is enough to try it |

---

## Setup

**1. Get the code**

```bat
git clone <your-repo-url> HeyArro
cd HeyArro
```

**2. Create a virtual environment**

```bat
python -m venv heyarro-env
heyarro-env\Scripts\python.exe -m pip install --upgrade pip
```

**3. Install the dependencies**

```bat
heyarro-env\Scripts\python.exe -m pip install -r requirements.txt
```

**4. Configure it**

```bat
copy .env.example .env
```

Open `.env` and set your own `GEMINI_API_KEY`. Every other setting already has a
sensible default, and each one is documented in `.env.example`.

**5. Run it**

```bat
heyarro-env\Scripts\python.exe -m app.main
```

Run it from the project root so the `app` package is importable. A tray icon appears
and the log prints `Clicky Assistant ready - hold Ctrl + Alt to record, transcribe,
capture and respond` (the tray, window titles and log lines still carry the prototype
name, "Clicky Assistant").

**6. Use it**

Hold `Ctrl + Alt`, ask your question out loud, release. Hey Arro records, transcribes,
captures the screen, thinks, and then teaches you on top of your own screen. Use the
tray icon for a status summary and to quit.

### A note on the virtual environment

`heyarro-env\Scripts\activate.bat` hard-codes the absolute path of the environment it
was created in. If you ever move or rename the folder, activation silently breaks and
`python` falls back to your system interpreter - which then fails with
`ModuleNotFoundError: No module named 'PySide6'`. Either delete and recreate the
environment, or simply call the interpreter directly, as every command above does.

---

## Configuration

All settings live in `.env` (see `.env.example` for the annotated list). Real
environment variables win over the file, so a one-off run can override anything:

```bat
set LLM_PROVIDER=mock && heyarro-env\Scripts\python.exe -m app.main
```

The ones you are most likely to touch:

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` for real answers, `mock` for an offline echo |
| `GEMINI_API_KEY` | *(empty)* | your personal key - the only secret the app needs |
| `GEMINI_MODEL` | `gemini-3.8-flash` | any vision-capable model your account can use |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`…`large-v3`; bigger is slower but more accurate |
| `TTS_PROVIDER` | `windows` | `windows` voice, `mock` for silence |
| `TEACHING_ENABLED` | `true` | `false` = voice only, no overlay, no grounding |
| `VISUAL_GROUNDING_ENABLED` | `true` | refine target positions with a second pass |
| `SCREENSHOT_MONITOR` | `1` | which monitor to capture (`0` = all combined) |

Every name also accepts a `CLICKY_`-prefixed alias from the original prototype.

---

## Running it without a key

You can try the whole app offline: with `LLM_PROVIDER=mock` and `TTS_PROVIDER=mock`,
the pipeline runs end to end with a canned response, no network and no voice.

The scripts below exercise real systems (the real coordinator, the real overlay, the
real screen capture), which is the quickest way to see what the teaching layer does:

```bat
:: one scripted multi-step visual lesson, silent (real overlay + real screen)
set TTS_PROVIDER=mock && heyarro-env\Scripts\python.exe -m app.teaching

:: just the voice, to check the Windows speech engine
heyarro-env\Scripts\python.exe -m app.tts "Hey Arro is ready."
```

## Tests

```bat
heyarro-env\Scripts\python.exe -m pytest -q
```

No API key, microphone or model download is required.

---

## Project layout

```
app/
  main.py             composition root: builds the services, tray and hotkey
  config.py           every setting, read from .env / the environment
  state.py            the five-state lifecycle and its valid transitions
  coordinator.py      drives one interaction end to end
  pipeline.py         pairs the transcript with the screenshot of one interaction
  audio/              microphone recording (sounddevice) -> 16 kHz WAV
  hotkey/             global Ctrl + Alt detection
  transcription/      provider contract + local faster-whisper implementation
  capture/            provider contract + mss screen capture
  llm/                vision-provider contract, response models, workers
                      (gemini + mock implemented, openai/claude are placeholders)
  teaching/           plans, difficulty, follow-ups, prompts, overlay,
                      primitives, coordinate mapping, execution sequence
  visual_grounding/   target refinement: crop -> close-up pass -> coordinate
                      mapping -> confidence -> fallback
  tts/                voice contract + Windows SAPI5 + mock + placeholder
  ui/                 tray, indicator, cursor companion, transcript caption
tests/                behaviour, no keys or hardware required
```

The layers do not leak into each other: only the Gemini module knows about the Gemini
SDK, only the overlay draws, only the grounding module refines, and the teaching
sequence only executes a plan that has already been prepared.

---

## Where your data goes

- **Audio** is written to `%TEMP%\clicky-recordings` and transcribed locally. It is
  never uploaded.
- **One screenshot per question** is written to `%TEMP%\clicky-screenshots` and **is
  sent to the configured vision provider** (Gemini) with your question. If you would
  rather not send screenshots anywhere, use `LLM_PROVIDER=mock`.
- **Memory:** the app remembers the lesson it just taught, in memory, so a follow-up
  can continue it. There is no account, no history, no learner profile and no
  telemetry.
- Both folders hold ordinary files you can delete at any time.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'PySide6'`**
You are running the system Python rather than the environment. Use
`heyarro-env\Scripts\python.exe -m app.main` from the project root (see the note on
virtual environments above).

**Holding Ctrl + Alt does nothing**
The global hotkey installs a low-level keyboard hook; some machines require the app to
run elevated for that. Check the tray status dialog first, and try running the terminal
as administrator. A GPU driver that also uses `Ctrl + Alt` (screen rotation, for
example) can swallow the combination.

**`GEMINI_API_KEY is not set` in the log**
The app falls back to the mock provider so it keeps running. Put your key in `.env`
(and remember to restart it).

**The first question takes a while**
The Whisper model downloads once on first use. Then each distinct visual target costs
one close-up refinement call (~10-15 s each), and *all* of it happens before the first
word is spoken - that is the design: no pauses between steps. Set
`VISUAL_GROUNDING_ENABLED=false` for a much faster, unrefined run.

**It narrates but nothing appears on screen**
Check that `TEACHING_ENABLED=true` and that the tray status reports an overlay service.

**It draws but says nothing**
Check `TTS_PROVIDER` and test the voice directly with
`python -m app.tts "hello"` (the Windows provider needs `pyttsx3` + SAPI5 voices).

**Something failed mid-question**
Errors are surfaced in a tray notification and the app returns to idle. The full
reason is in the console log - it usually names the provider or the missing key.

---

## Status and limitations

- A prototype, run from source: there is no installer, no packaging and no
  auto-update.
- Windows only, because of the global hotkey and the built-in voice.
- The OpenAI and Claude providers are registered placeholders that report "not
  implemented yet", and the ElevenLabs voice is a placeholder too.
- No clicking, typing or other computer-use: Hey Arro explains what is on screen and
  never operates your machine.
- The tray and window titles still read "Clicky Assistant", the name of the prototype
  this was built from.

## License

No license file is included yet - add one before distributing or publishing this
project.
