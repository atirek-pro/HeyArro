# Hey Arro

Hey Arro is a small Windows desktop assistant that sits next to your cursor.

Hold a hotkey, ask a question out loud, and Hey Arro transcribes it locally, takes
one screenshot of your screen, asks a vision model, and shows the answer in a
bubble next to the cursor - so a question like "what is this window for?" is
answered about what you are actually looking at.

```
Ctrl + Alt held      ->  record the microphone
Ctrl + Alt released  ->  transcribe locally (faster-whisper) + capture the screen
                     ->  ask the vision model with the question, the screenshot
                         and a depth instruction
                     ->  show the answer in a bubble next to the cursor
                     ->  idle, waiting for the next question
```

The whole input of every interaction - the transcript, the screenshot's size, the
depth and the exact instruction sent - is logged before the request goes out, so
anything you see on screen can be traced back to what the model was given.

---

## Features

**Push-to-talk, anywhere in Windows**
Hold `Ctrl + Alt` while Hey Arro is unfocused, speak, release. One interaction per
press; the next press starts a new one.

**Your voice stays on your machine**
Speech is transcribed locally with faster-whisper (CPU `int8` by default, model
size configurable, language auto-detected). Audio is never uploaded.

**It answers about what you are looking at**
One screenshot per question (monitor selectable) is attached to the request, so the
answer can name the window, the file, the error or the button you are asking about.

**The answer appears where you are looking**
A frameless, click-through bubble shows what was heard and then the answer, next to
the cursor, and disappears after a few seconds. It never takes focus and never
covers your work.

**Depth follows what you asked for**
`beginner`, `intermediate` and `advanced` are decided from *your wording* ("explain
this like I'm a beginner", "give me the technical implementation details"), never
from guesses about you - a technically hard question does not make you an advanced
learner. The level changes the words of the answer and costs no extra model call.

**One model call per question**
Nothing else is requested: no classification call, no second pass, no follow-up.
Press, ask, read.

**Runs without any keys at all**
`LLM_PROVIDER=mock` answers offline from a canned response, so the whole flow can be
exercised with no API key and no network.

**Everything is tested**
171 tests cover the state machine, the coordination, the prompt and request
contract, the answer schema, transcription, capture and the hotkey tracker. None of
them needs an API key, a microphone or a real model.

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (the global hotkey is Windows-specific) |
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
and the log prints `Hey Arro ready - hold Ctrl + Alt to ask`.

**6. Use it**

Hold `Ctrl + Alt`, ask your question out loud, release. The bubble shows what was
heard, then the answer. Use the tray icon for a status summary and to quit.

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
| `TRANSCRIPT_DISPLAY_MS` | `4000` | how long the bubble stays on screen |
| `SCREENSHOT_MONITOR` | `1` | which monitor to capture (`0` = all combined) |
| `GEMINI_TIMEOUT_SECONDS` | `60` | how long to wait before reporting a failure |

Every name also accepts a `CLICKY_`-prefixed alias from the original prototype.

### Running it without a key

`LLM_PROVIDER=mock` runs the whole flow offline with a canned answer, which is the
quickest way to check the hotkey, the microphone, the capture and the bubble.

## Tests

```bat
heyarro-env\Scripts\python.exe -m pytest -q
```

No API key, microphone or model download is required.

---

## Project layout

```
app/
  main.py             composition root: builds the services, tray, hotkey and bubble
  config.py           every setting, read from .env / the environment
  state.py            the five-state lifecycle and its valid transitions
  coordinator.py      drives one interaction: record, transcribe, capture, ask, show
  pipeline.py         pairs the transcript with the screenshot of one interaction
  audio/              microphone recording (sounddevice) -> 16 kHz WAV
  hotkey/             global Ctrl + Alt detection
  transcription/      provider contract + local faster-whisper implementation
  capture/            provider contract + mss screen capture
  llm/                the provider contract, the request and answer schema,
                      the prompt, workers (gemini + mock; openai/claude placeholders)
  teaching/           how deeply to answer, decided from the wording of the request
  ui/                 tray, indicator, cursor companion, answer bubble
tests/                behaviour, no keys or hardware required
```

The layers do not leak into each other: only the Gemini module knows about the
Gemini SDK, only the coordinator owns the lifecycle, only the prompt module writes
prompt text, and only the UI draws.

---

## Where your data goes

- **Audio** is written to `%TEMP%\clicky-recordings` and transcribed locally. It is
  never uploaded.
- **One screenshot per question** is written to `%TEMP%\clicky-screenshots` and **is
  sent to the configured vision provider** (Gemini) with your question. If you would
  rather not send screenshots anywhere, use `LLM_PROVIDER=mock`.
- **Memory:** each interaction is independent. There is no history, no account, no
  profile and no telemetry.
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
Every question will fail with a clear message until it is configured. Put your key in
`.env` and restart.

**`[Input]` is logged but no answer appears**
Look at the log line after it: a rejected answer is reported as an error (the tray also
shows it) instead of being displayed. The most common cause is a wrong or expired key.

**The answer is refused by the model**
Vision models decline some content. The error text names it; nothing is shown in the
bubble.

**The first question takes a while**
The Whisper model is downloaded once on first use, and each answer is one round trip to
the model (measured at roughly 10-15 s with `gemini-3.8-flash`). Transcription and
capture run in parallel with nothing else, so the wait is the model's.

---

## Status and limitations

- A prototype, run from source: there is no installer, no packaging and no
  auto-update.
- Windows only, because of the global hotkey.
- The answer is plain text in a bubble. Earlier versions also spoke the answer,
  drew a pointer and highlights over the screen, and continued the lesson on
  follow-up questions; those stages have been removed, and the model is now asked
  for the answer alone.
- The OpenAI and Claude providers are registered placeholders that report "not
  implemented yet".
- No clicking, typing or other computer-use: Hey Arro only reads the screen.
- The model's answer is shown as it comes back: there is no second pass, so the
  wording is the model's.

## License

No license file is included yet - add one before distributing or publishing this
project.
