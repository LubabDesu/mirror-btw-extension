"""Ask a side question with /btw while the main task keeps running."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import uuid
from collections.abc import Iterable, Sequence
from typing import override

from prompt_toolkit.layout import containers, controls

from mirror.cli import (
  actions,
  agents,
  commands,
  effects,
  extension,
  rpc_client,
  workspace_application,
)
from mirror.cli import state as _state
from mirror.protocol import agent_pb2

try:
  from mirror.cli.components import panel
except ImportError:  # Mirror without the panel region.
  panel = None


class _Unsupported:
  """Stands in for a panel action that this Mirror does not define."""

  __match_args__ = ('delta',)


_ClosePanel = getattr(actions, 'ClosePanel', _Unsupported)
_ScrollPanel = getattr(actions, 'ScrollPanel', _Unsupported)
_CopyPanel = getattr(actions, 'CopyPanel', _Unsupported)
_PANEL_REGION = getattr(extension.RenderRegion, 'PANEL', None)
# Without the panel region the answer shows in a widget above the prompt.
_PANEL_SUPPORTED = (
  _PANEL_REGION is not None
  and panel is not None
  and _Unsupported not in (_ClosePanel, _ScrollPanel, _CopyPanel)
)

_MAX_HISTORY_CHARS = 60_000
# Rows of answer text in the panel; the frame adds a title and a hint row.
_BODY_ROWS = 10
_HINT = 'j/k scroll · c copy · esc close'
# The widget cannot scroll, so it gets more rows.
_WIDGET_ROWS = 20
_WIDGET_HINT = '/btw alone dismisses'
_TIMEOUT_SECONDS = 300
_SYSTEM_PROMPT = """\
You answer a quick side question about the conversation below.
Answer from the conversation alone, briefly. Do not call any tool.

<conversation>
{history}
</conversation>"""


@dataclasses.dataclass(frozen=True)
class _Ask(actions.ExtensionAction):
  question: str


@dataclasses.dataclass(frozen=True)
class _Chunk(actions.ExtensionAction):
  run: int
  text: str


@dataclasses.dataclass(frozen=True)
class _Final(actions.ExtensionAction):
  run: int
  text: str


@dataclasses.dataclass(frozen=True)
class _Failed(actions.ExtensionAction):
  run: int
  message: str


@dataclasses.dataclass(frozen=True)
class _Run:
  run: int
  question: str


@dataclasses.dataclass(frozen=True)
class _Copy:
  text: str


type _Effect = _Run | _Copy | str  # 'cancel' stops the side agent.


@dataclasses.dataclass(frozen=True)
class BtwState:
  """The side question on screen; `run` ignores stale updates."""

  phase: str = 'idle'  # idle | asking | done | error
  run: int = 0
  question: str = ''
  answer: str = ''
  scroll: int = 0  # First visible wrapped line once the answer is done.


def history_text(records: Iterable[agent_pb2.AgentMessage]) -> str:
  """Render the model-visible conversation as text, newest part kept."""
  lines: list[str] = []
  for record in records:
    payload = record.WhichOneof('payload')
    if payload == 'context_window' and record.context_window.HasField(
      'compaction'
    ):
      # A compaction replaces everything before it with a summary and recent turns.
      compaction = record.context_window.compaction
      lines = []
      if compaction.HasField('summary'):
        lines.append(f'[Earlier summary]\n{compaction.summary}')
      lines.extend(_chat_lines(compaction.recent))
    elif payload == 'chat':
      lines.extend(_chat_lines((record,)))
  text = '\n\n'.join(lines)
  return text[-_MAX_HISTORY_CHARS:]


def _chat_lines(records: Iterable[agent_pb2.AgentMessage]) -> list[str]:
  lines: list[str] = []
  for record in records:
    if record.WhichOneof('payload') != 'chat':
      continue
    chat = record.chat
    match chat.WhichOneof('message'):
      case 'user' if chat.user.content.strip():
        lines.append(f'User: {chat.user.content.strip()}')
      case 'assistant' if chat.assistant.content.strip():
        lines.append(f'Assistant: {chat.assistant.content.strip()}')
  return lines


def _wrap(text: str, columns: int) -> list[str]:
  """Split text into the rows it takes at the given width."""
  width = max(1, columns)
  rows: list[str] = []
  for line in text.splitlines():
    rows.extend(line[i : i + width] for i in range(0, len(line), width))
    if not line:
      rows.append('')
  return rows


class Btw(extension.Extension[BtwState, _Effect, None]):
  """Answer a side question from the conversation, without disturbing it."""

  def __init__(self) -> None:
    self._tasks: dict[extension.Workspace, asyncio.Task[None]] = {}
    # Never reused, so a state reset cannot make an old run look current.
    self._runs = itertools.count(1)

  @classmethod
  @override
  def id(cls) -> str:
    return 'mirror-btw-extension'

  @staticmethod
  @override
  def initial_state() -> BtwState:
    return BtwState()

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'btw',
        'Ask a side question; /btw alone dismisses the answer',
        self._handle_command,
        arguments=commands.OptionalArguments('[question]'),
        available_when_busy=True,
      ),
    )

  def _handle_command(
    self, context: commands.CommandContext, arguments: str
  ) -> actions.Action:
    del context
    question = arguments.strip()
    return actions.RoutedExtensionAction(self.id(), _Ask(question))

  @override
  def reduce(
    self, global_state: _state.State, action: actions.Action
  ) -> tuple[
    BtwState, tuple[effects.CoreEffect | extension.PrivateEffect[_Effect], ...]
  ]:
    current = self.require_state(global_state)
    match action:
      case _ClosePanel():
        if current.phase == 'idle':
          return current, ()
        return BtwState(run=next(self._runs)), (self.effect('cancel'),)
      case _ScrollPanel(delta) if current.phase != 'asking':
        total = len(_wrap(current.answer, global_state.layout.columns))
        top = min(max(0, current.scroll + delta), max(0, total - _BODY_ROWS))
        return dataclasses.replace(current, scroll=top), ()
      case _CopyPanel():
        if current.phase == 'idle':
          return current, ()
        return current, (self.effect(_Copy(current.answer)),)
    if not isinstance(action, actions.RoutedExtensionAction):
      return current, ()
    match action.action:
      case _Ask(''):  # /btw alone dismisses
        return BtwState(run=next(self._runs)), (self.effect('cancel'),)
      case _Ask(question):
        run = next(self._runs)
        return BtwState('asking', run, question), (
          self.effect(_Run(run, question)),
        )
      case _Chunk(run, text) if (
        run == current.run and current.phase == 'asking'
      ):
        return dataclasses.replace(current, answer=current.answer + text), ()
      case _Final(run, text) if run == current.run:
        # The answer stays in the panel; the main chat is not touched.
        return dataclasses.replace(
          current, phase='done', answer=text, scroll=0
        ), ()
      case _Failed(run, message) if run == current.run:
        return dataclasses.replace(current, phase='error', answer=message), ()
      case _:
        return current, ()

  @override
  def render_region(self) -> extension.RenderRegion:
    if _PANEL_SUPPORTED:
      return _PANEL_REGION
    return extension.RenderRegion.WIDGET_ABOVE_PROMPT

  @override
  def render(
    self, global_state: _state.State, extension_state: BtwState
  ) -> containers.AnyContainer | None:
    if extension_state.phase == 'idle':
      return None
    rows = _BODY_ROWS if _PANEL_SUPPORTED else _WIDGET_ROWS
    lines = _wrap(extension_state.answer, global_state.layout.columns)
    if extension_state.phase == 'asking':
      shown = lines[-rows:]  # Follow the stream.
    else:
      shown = lines[extension_state.scroll : extension_state.scroll + rows]
    failed = extension_state.phase == 'error'
    style = 'class:ui.error' if failed else 'class:ui.text'
    body = containers.Window(
      controls.FormattedTextControl([(style, '\n'.join(shown))]),
      dont_extend_height=True,
    )
    title = (
      f'/btw failed: {extension_state.question}'
      if failed
      else f'/btw {extension_state.question}'
    )
    if _PANEL_SUPPORTED:
      return panel.frame(title, body, _HINT, _BODY_ROWS)
    return containers.HSplit([
      containers.Window(
        controls.FormattedTextControl([('class:ui.accent bold', f' {title}')]),
        height=1,
      ),
      body,
      containers.Window(
        controls.FormattedTextControl([('class:ui.dim', f' {_WIDGET_HINT}')]),
        height=1,
      ),
    ])

  @override
  async def on_unload(self, context: None) -> None:
    for workspace in tuple(self._tasks):
      await self._cancel(workspace)

  async def _cancel(self, workspace: extension.Workspace) -> None:
    # Loop: another handler can start a task while this one awaits cleanup.
    while (task := self._tasks.get(workspace)) is not None:
      # A second cancel would interrupt client.close().
      if not task.cancelling():
        task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      if self._tasks.get(workspace) is task:
        del self._tasks[workspace]

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _Effect
  ) -> None:
    workspace = context.workspace
    if isinstance(effect, _Copy):  # Copying must not cancel a running task.
      try:
        workspace_application.copy_text(effect.text)
      except (ValueError, OSError) as error:
        context.dispatch(
          actions.ShowText(f'Copy failed: {error}', style='ui.error')
        )
      return
    await self._cancel(workspace)
    if not isinstance(effect, _Run):
      return
    # No await between the loop above and this assignment, so only one task runs.
    task = asyncio.create_task(self._ask(context, effect))
    self._tasks[workspace] = task
    try:
      await task
    except asyncio.CancelledError:
      # A newer /btw or a dismiss cancelled this run; it owns the screen now.
      # Re-raise when our own caller was cancelled.
      current = asyncio.current_task()
      if current is not None and current.cancelling():
        raise
    except (NotImplementedError, OSError, RuntimeError, TimeoutError) as error:
      detail = (
        f'timed out after {_TIMEOUT_SECONDS}s'
        if isinstance(error, TimeoutError) and not str(error)
        else str(error) or type(error).__name__
      )
      self.dispatch(
        context, _Failed(effect.run, f'Side question failed: {detail}')
      )
    finally:
      if self._tasks.get(workspace) is task:
        del self._tasks[workspace]

  async def _ask(self, context: extension.Context, effect: _Run) -> None:
    events: asyncio.Queue[agent_pb2.SessionMessage | None] = asyncio.Queue()

    async def on_error(error: str) -> None:
      del error  # Transport errors can carry provider payloads.
      events.put_nowait(None)

    conversation = context.get_state().session.conversation_id
    prompt = _SYSTEM_PROMPT.format(history=history_text(context.transcript()))
    async with (
      asyncio.timeout(_TIMEOUT_SECONDS),
      context.create_agent(
        agents.Options(system_prompt=prompt), events.put_nowait, on_error
      ) as client,
    ):
      # Client has no public conversation ID yet; core could expose one.
      session_id: str = client._conversation_id  # pyright: ignore[reportPrivateUsage]
      # Manual approval is a partial guard: tools that need no approval still run.
      client.send(
        agent_pb2.ClientMessage(
          session_id=session_id,
          message_id=uuid.uuid4().hex,
          set_approval_mode=agent_pb2.SetApprovalMode(
            mode=agent_pb2.APPROVAL_MODE_HITL
          ),
        )
      )
      client.enqueue(
        agent_pb2.AgentMessage(
          id=uuid.uuid4().hex,
          chat=agent_pb2.ChatRecord(
            user=agent_pb2.UserMessage(content=effect.question),
            success=True,
          ),
        )
      )
      await self._stream(
        context, effect.run, client, session_id, events, conversation
      )

  async def _stream(
    self,
    context: extension.Context,
    run: int,
    client: rpc_client.Client,
    session_id: str,
    events: asyncio.Queue[agent_pb2.SessionMessage | None],
    conversation: str,
  ) -> None:
    answer = ''
    while True:
      event = await events.get()
      # The side agent outlives a session switch; drop what it says after it.
      if context.get_state().session.conversation_id != conversation:
        return
      if event is None:
        raise RuntimeError('agent connection failed')
      payload = event.WhichOneof('payload')
      if payload == 'hitl_request':
        client.send(
          agent_pb2.ClientMessage(
            session_id=session_id,
            message_id=uuid.uuid4().hex,
            hitl_response=agent_pb2.HITLResponse(
              call_id=event.hitl_request.call.id,
              approved=False,
              source=agent_pb2.APPROVAL_SOURCE_MANUAL,
            ),
          )
        )
      elif payload == 'input_nack':
        nack = event.input_nack
        raise RuntimeError(
          f'agent rejected input {nack.message_id[:8]} '
          f'({agent_pb2.InputNackReason.Name(nack.reason)}): {nack.error}'
        )
      elif payload == 'agent_message':
        record = event.agent_message
        if record.chat.WhichOneof('message') == 'assistant':
          assistant = record.chat.assistant
          # Limit notices are synthetic records, not model completions.
          answer = (
            assistant.content or assistant.refusal
            if record.HasField('inference')
            and record.chat.success
            and not record.chat.truncated
            and not assistant.tool_calls
            else ''
          )
      elif payload == 'runtime_delta':
        delta = event.runtime_delta.WhichOneof('payload')
        if delta == 'stream_content':
          self.dispatch(
            context, _Chunk(run, event.runtime_delta.stream_content.content)
          )
        elif delta == 'turn_error':
          raise RuntimeError('agent turn failed')
        elif delta == 'turn_interrupted':
          raise RuntimeError('agent turn was interrupted')
        elif delta == 'turn_completed':
          if not answer.strip():
            raise RuntimeError('agent ended without an answer')
          self.dispatch(context, _Final(run, answer))
          return
