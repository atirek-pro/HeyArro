# Hey Arro

Hey Arro is a small Windows background assistant for asking a vision model about
your screen, by voice.

Hold a hotkey, ask a question out loud, and Hey Arro transcribes it locally, takes one
screenshot, sends both to the model, and writes the answer to the log. There is no
window, no bubble and no state machine: the log is the interface.

```
Ctrl + Alt held      ->  record the microphone
Ctrl + Alt released  ->  transcribe locally (faster-whisper) + capture the screen
                     ->  log what is about to be sent
                     ->  one model call: the question + the screenshot
                     ->  log the answer
                     ->  ready for the next question
```

The log of one interaction looks like this:

```
[Input] transcript: 'what is on my screen?'
[Input] screenshot: 1920x1080, 245579 bytes, monitor 1
[Input] sending to GeminiVisionProvider (model=gemini-3.8-flash)
Response [tone=instructional, 268 characters]
[Output] The main window is a browser playing a video about inference engines...
```

---

## Features

**Push-to-talk, anywhere in Windows**
Hold `Ctrl + Alt` while Hey Arro is unfocused, speak, release. One interaction per
press; the next press starts a new one.

**Your voice stays on your machine**
Speech is transcribed locally with faster-whisper (CPU `int8` by default, model size
configurable, language auto-detected). Audio is never uploaded.

**It asks about what you are looking at**
One screenshot per question (monitor selectable) is attached to the request, so the
answer can name the window, the file, the error or the button you are asking about.

**The answer is logged, not displayed**
Nothing is drawn, spoken or stored: the answer goes to the log as it arrives, next to
the input that produced it. That makes the app unobtrusive, and makes what the model
was given and what it said completely traceable.

**One interaction at a time**
A press is ignored while a recording or a question is still in flight, so two runs can
never interleave in the log.

**One model call per question**
No classification call, no second pass, no follow-up. Press, ask, read the log.

**Runs without any keys at all**
`LLM_PROVIDER=mock` answers offline from a canned response, so the whole flow can be
exercised with no API key and no network.

**Everything is tested**
112 tests cover the coordination, the prompt and request contract, the answer schema,
transcription, capture and the hotkey tracker. None of them needs an API key, a
microphone or a real model.

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (the global hotkey is Windows-specific) |
| Python | 3.12 (developed on 3.12.10) |
| Hardware | a microphone |
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

Run it from the project root so the `app` package is importable. It prints
`Hey Arro ready - hold Ctrl + Alt to ask` and then keeps running in the background; the
answers appear in that console, so keep the window visible.

**6. Use it**

Hold `Ctrl + Alt`, ask your question out loud, release, and read the answer in the
log. The tray icon shows a status summary and quits the app.

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

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` for real answers, `mock` for an offline echo |
| `GEMINI_API_KEY` | *(empty)* | your personal key - the only secret the app needs |
| `GEMINI_MODEL` | `gemini-3.8-flash` | any vision-capable model your account can use |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`…`large-v3`; bigger is slower but more accurate |
| `SCREENSHOT_MONITOR` | `1` | which monitor to capture (`0` = all combined) |
| `GEMINI_TIMEOUT_SECONDS` | `60` | how long to wait before reporting a failure |

Every name also accepts a `CLICKY_`-prefixed alias from the original prototype.

### Running it without a key

`LLM_PROVIDER=mock` runs the whole flow offline with a canned answer, which is the
quickest way to check the hotkey, the microphone, the capture and the logging.

## Tests

```bat
heyarro-env\Scripts\python.exe -m pytest -q
```

No API key, microphone or model download is required.

---

## Project layout

```
app/
  main.py             composition root: builds the services, the hotkey and the tray
  config.py           every setting, read from .env / the environment
  coordinator.py      drives one interaction: record, transcribe, capture, ask, log
  pipeline.py         pairs the transcript with the screenshot of one interaction
  audio/              microphone recording (sounddevice) -> 16 kHz WAV
  hotkey/             global Ctrl + Alt detection
  transcription/      provider contract + local faster-whisper implementation
  capture/            provider contract + mss screen capture
  llm/                the provider contract, the request and answer schema,
                      the prompt, workers (gemini + mock; openai/claude placeholders)
  ui/                 the tray icon (status and quit) - the only UI left
tests/                behaviour, no keys or hardware required
```

The layers do not leak into each other: only the Gemini module knows about the Gemini
SDK, only the coordinator owns the lifecycle, only the prompt module writes prompt
text, and the application never renders the answer.

---

## Where your data goes

- **Audio** is written to `%TEMP%\clicky-recordings` and transcribed locally. It is
  never uploaded.
- **One screenshot per question** is written to `%TEMP%\clicky-screenshots` and **is
  sent to the configured vision provider** (Gemini) with your question. If you would
  rather not send screenshots anywhere, use `LLM_PROVIDER=mock`.
- **Memory:** each interaction is independent. The answer is logged and dropped: no
  history, no account, no profile, no telemetry.
- Both folders hold ordinary files you can delete at any time.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'PySide6'`**
You are running the system Python rather than the environment. Use
`heyarro-env\Scripts\python.exe -m app.main` from the project root (see the note on
virtual environments above).

**Holding Ctrl + Alt does nothing**
The global hotkey installs a low-level keyboard hook; some machines require the app to
run elevated for that. Check the tray status first, and try running the terminal as
administrator. A GPU driver that also uses `Ctrl + Alt` (screen rotation, for example)
can swallow the combination.

**Nothing appears in the log**
The console must stay open: `logging` writes to it. If the tray shows a warning, the
reason is in the log line just before it.

**`GEMINI_API_KEY is not set` in the log**
Every question fails with a clear message until it is configured. Put your key in
`.env` and restart.

**The first question takes a while**
The Whisper model is downloaded once on first use, and each answer is one round trip to
the model (measured at roughly 10-15 s with `gemini-3.8-flash`).

---

## Status and limitations

- A prototype, run from source: there is no installer, no packaging and no
  auto-update.
- Windows only, because of the global hotkey.
- The application displays nothing by design: the answer is written to the log, which
  is why the launcher has to keep its console open.
- Earlier versions spoke the answer, drew a pointer and highlights over the screen,
  showed a bubble next to the cursor, tracked an application state machine and adapted
  the depth of the explanation. Those stages have all been removed.
- The OpenAI and Claude providers are registered placeholders that report "not
  implemented yet".
- No clicking, typing or other computer-use: Hey Arro only reads the screen.

## License

No license file is included yet - add one before distributing or publishing this
project.
