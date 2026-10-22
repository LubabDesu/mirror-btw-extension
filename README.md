# Mirror /btw extension

Ask a side question with `/btw <question>` while the main Mirror task keeps
running. A side agent answers from the conversation alone. It does not call
tools and it does not touch the main chat.

## Install and enable

Review `extension.py`, then enter this command **inside Mirror**:

```text
/extensions-add https://github.com/<your-user>/mirror-btw-extension
```

Mirror downloads the repository and asks for consent. Select **Enable**.

**Trust boundary:** extensions execute unsandboxed Python locally with your
account's filesystem, network, and credential access. Only enable reviewed code.

## Use

- `/btw <question>` starts a side question.
- `/btw` alone dismisses the answer and cancels a running side question.

## Two display modes

The extension picks the mode when it loads.

| Mirror | Where the answer shows | Keys |
|---|---|---|
| Has `RenderRegion.PANEL` | A panel that replaces the prompt box | `j`/`k` scroll, `c` copy, `Esc` close |
| Stock `reflection-mirror` 0.1.0 | A widget above the prompt (20 rows) | `/btw` alone dismisses |

The panel mode needs a Mirror build with the generic `PANEL` region. Stock
Mirror does not have it yet. The fallback needs no core change.

## Limits

- The side agent sees chat text only. It cannot see tool output.
- It is asked not to call tools. Manual approval mode auto-denies tools that
  need approval, but tools that need no approval can still run.
- It runs locally only. Remote workspaces are not supported.
- A side task stops after 300 seconds.

## Test

Install Mirror and pytest in a virtual environment, then run:

```sh
python -m pytest extension_test.py
```

Tests for the panel actions skip on a Mirror without the panel region.

## License

[Apache-2.0](LICENSE).
