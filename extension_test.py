"""Tests for the /btw extension's pure logic."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import pathlib
import sys
import types
from typing import Any

import pytest

from mirror.cli import actions
from mirror.protocol import agent_pb2

_SPEC = importlib.util.spec_from_file_location(
  'btw_extension', pathlib.Path(__file__).with_name('extension.py')
)
assert _SPEC is not None and _SPEC.loader is not None
btw = importlib.util.module_from_spec(_SPEC)
# dataclasses resolve annotations through sys.modules
sys.modules[_SPEC.name] = btw
_SPEC.loader.exec_module(btw)


needs_panel = pytest.mark.skipif(
  not hasattr(actions, 'ClosePanel'),
  reason='this Mirror has no panel region',
)


def _chat(user: str | None = None, assistant: str | None = None) -> Any:
  chat = agent_pb2.ChatRecord(success=True)
  if user is not None:
    chat.user.content = user
  if assistant is not None:
    chat.assistant.content = assistant
  return agent_pb2.AgentMessage(id='x', chat=chat)


def _reduce(state: Any, action: Any) -> Any:
  ext = btw.Btw()
  return ext.reduce(
    _global(state), actions.RoutedExtensionAction(btw.Btw.id(), action)
  )


def _global(state: Any, columns: int = 80) -> Any:
  return types.SimpleNamespace(
    extensions={btw.Btw.id(): state}, layout=types.SimpleNamespace(columns=columns)
  )


def _core(state: Any, action: Any, columns: int = 80) -> Any:
  return btw.Btw().reduce(_global(state, columns), action)


def test_history_renders_user_and_assistant_turns() -> None:
  text = btw.history_text([_chat(user='fix bug'), _chat(assistant='done')])
  assert text == 'User: fix bug\n\nAssistant: done'


def test_history_restarts_from_compaction() -> None:
  compaction = agent_pb2.AgentMessage(
    id='c',
    context_window=agent_pb2.ContextWindowRecord(
      compaction=agent_pb2.Compaction(
        summary='old work', recent=[_chat(user='recent')]
      )
    ),
  )
  text = btw.history_text([
    _chat(user='before'),
    compaction,
    _chat(user='after'),
  ])
  assert 'before' not in text
  assert text.splitlines()[0] == '[Earlier summary]'
  assert (
    'old work' in text and 'recent' in text and text.endswith('User: after')
  )


def test_history_keeps_newest_text_within_budget() -> None:
  long = 'a' * (btw._MAX_HISTORY_CHARS + 10)
  text = btw.history_text([_chat(user=long), _chat(assistant='tail')])
  assert len(text) == btw._MAX_HISTORY_CHARS
  assert text.endswith('Assistant: tail')


def test_ask_starts_run_and_emits_effect() -> None:
  state, emitted = _reduce(btw.BtwState(run=2), btw._Ask('why?'))
  assert (state.phase, state.question) == ('asking', 'why?')
  assert state.run != 2
  assert emitted[0].payload == btw._Run(state.run, 'why?')


@needs_panel
def test_run_ids_stay_monotonic_across_state_resets() -> None:
  ext = btw.Btw()
  first, _ = ext.reduce(
    _global(btw.BtwState()), actions.RoutedExtensionAction(btw.Btw.id(), btw._Ask('a'))
  )
  # A session switch resets the state to run 0; the old run must stay stale.
  second, _ = ext.reduce(
    _global(btw.BtwState()), actions.RoutedExtensionAction(btw.Btw.id(), btw._Ask('b'))
  )
  assert second.run > first.run
  closed, _ = ext.reduce(_global(second), actions.ClosePanel())
  assert closed.run > second.run


@needs_panel
def test_scroll_ignored_while_asking_and_reset_on_final() -> None:
  answer = '\n'.join(f'line {i}' for i in range(btw._BODY_ROWS + 5))
  asking = btw.BtwState('asking', run=1, question='q', answer=answer)
  same, _ = _core(asking, actions.ScrollPanel(3))
  assert same.scroll == 0
  scrolled = btw.BtwState('done', run=1, question='q', answer=answer, scroll=4)
  done, _ = _reduce(
    btw.BtwState('asking', run=1, question='q', scroll=4), btw._Final(1, 'x')
  )
  assert done.scroll == 0 and scrolled.scroll == 4


def test_stale_chunk_is_ignored() -> None:
  state = btw.BtwState('asking', run=3, question='q')
  same, _ = _reduce(state, btw._Chunk(2, 'old'))
  assert same.answer == ''
  grown, _ = _reduce(state, btw._Chunk(3, 'new'))
  assert grown.answer == 'new'


def test_final_replaces_preview_and_leaves_main_chat_untouched() -> None:
  state = btw.BtwState('asking', run=1, question='q', answer='par')
  done, emitted = _reduce(state, btw._Final(1, 'full answer'))
  assert (done.phase, done.answer) == ('done', 'full answer')
  assert emitted == ()


def test_dismiss_clears_and_invalidates_running_task() -> None:
  state = btw.BtwState('asking', run=4, question='q', answer='x')
  cleared, emitted = _reduce(state, btw._Ask(''))
  assert cleared.phase == 'idle' and cleared.run != 4
  assert emitted[0].payload == 'cancel'
  late, _ = _reduce(cleared, btw._Chunk(4, 'late'))
  assert late.answer == ''


def test_command_without_question_dismisses() -> None:
  ext = btw.Btw()
  action = ext._handle_command(None, '  ')  # type: ignore[arg-type]
  assert action.action == btw._Ask('')


@needs_panel
def test_close_panel_resets_state_and_cancels() -> None:
  state = btw.BtwState('asking', run=4, question='q', answer='x')
  cleared, emitted = _core(state, actions.ClosePanel())
  assert cleared.phase == 'idle' and cleared.run != 4
  assert emitted[0].payload == 'cancel'


@needs_panel
def test_close_panel_idle_is_noop() -> None:
  state = btw.BtwState(run=2)
  same, emitted = _core(state, actions.ClosePanel())
  assert same == state and emitted == ()


@needs_panel
def test_scroll_panel_clamps_at_zero_and_last_line() -> None:
  answer = '\n'.join(f'line {i}' for i in range(btw._BODY_ROWS + 5))
  state = btw.BtwState('done', run=1, question='q', answer=answer)
  up, _ = _core(state, actions.ScrollPanel(-3))
  assert up.scroll == 0
  down, _ = _core(state, actions.ScrollPanel(100))
  assert down.scroll == 5
  mid, _ = _core(down, actions.ScrollPanel(-2))
  assert mid.scroll == 3


@needs_panel
def test_scroll_panel_counts_wrapped_lines() -> None:
  answer = 'x' * (
    10 * (btw._BODY_ROWS + 2)
  )  # One logical line, wraps at 10 columns.
  state = btw.BtwState('done', run=1, question='q', answer=answer)
  down, _ = _core(state, actions.ScrollPanel(100), columns=10)
  assert down.scroll == 2


@needs_panel
def test_scroll_panel_short_answer_stays_zero() -> None:
  state = btw.BtwState('done', run=1, question='q', answer='short')
  same, _ = _core(state, actions.ScrollPanel(5))
  assert same.scroll == 0


def test_render_none_when_idle() -> None:
  assert btw.Btw().render(_global(btw.BtwState()), btw.BtwState()) is None


def _texts(container: Any) -> list[str]:
  found: list[str] = []
  content = getattr(container, 'content', None)
  if content is not None and hasattr(content, 'text'):
    found.append(''.join(part[1] for part in content.text))
  for child in container.get_children():
    found.extend(_texts(child))
  return found


@needs_panel
def test_render_panel_region() -> None:
  from mirror.cli import extension

  ext = btw.Btw()
  assert ext.render_region() is extension.RenderRegion.PANEL
  state = btw.BtwState('done', run=1, question='q', answer='a\nb')
  texts = _texts(ext.render(_global(state), state))
  assert texts[0].strip() == '/btw q'
  assert 'a\nb' in texts
  assert texts[-1].strip() == btw._HINT


@needs_panel
def test_copy_panel_effect_contains_answer() -> None:
  state = btw.BtwState('done', run=1, question='q', answer='the answer')
  same, emitted = _core(state, actions.CopyPanel())
  assert same == state
  assert emitted[0].payload == btw._Copy('the answer')


def _copy_context(dispatched: list[Any]) -> tuple[Any, Any]:
  return btw.Btw(), types.SimpleNamespace(
    workspace='w', dispatch=dispatched.append
  )


def test_copy_effect_calls_copy_text(monkeypatch: Any) -> None:
  copied: list[str] = []
  monkeypatch.setattr(btw.workspace_application, 'copy_text', copied.append)
  dispatched: list[Any] = []
  ext, context = _copy_context(dispatched)
  asyncio.run(ext.handle_effect(context, btw._Copy('hi')))
  assert copied == ['hi'] and dispatched == []


@pytest.mark.parametrize(
  ('error', 'text'),
  [
    (
      ValueError('Text exceeds the terminal clipboard limit of 75 KB.'),
      'x' * 80_000,
    ),
    (OSError('no tty'), 'x'),
  ],
)
def test_copy_failure_reports_in_transcript(
  monkeypatch: Any, error: Exception, text: str
) -> None:
  def broken(text: str) -> None:
    raise error

  monkeypatch.setattr(btw.workspace_application, 'copy_text', broken)
  dispatched: list[Any] = []
  ext, context = _copy_context(dispatched)
  asyncio.run(ext.handle_effect(context, btw._Copy(text)))
  assert isinstance(dispatched[0], actions.ShowText)
  assert dispatched[0].text == f'Copy failed: {error}'


def _title(state: Any) -> str:
  container = btw.Btw().render(_global(state), state)
  return container.get_children()[0].content.text[0][1]


def test_error_phase_title_differs_from_done() -> None:
  done = btw.BtwState('done', run=1, question='q', answer='a')
  failed = btw.BtwState('error', run=1, question='q', answer='boom')
  assert _title(done) != _title(failed)
  assert 'failed' in _title(failed)


def test_timeout_message_is_not_empty() -> None:
  ext = btw.Btw()
  dispatched: list[Any] = []
  ext.dispatch = lambda context, action: dispatched.append(action)  # type: ignore[method-assign]

  async def timed_out(context: Any, effect: Any) -> None:
    raise TimeoutError()

  ext._ask = timed_out  # type: ignore[method-assign]
  context = types.SimpleNamespace(workspace='w')
  asyncio.run(ext.handle_effect(context, btw._Run(1, 'q')))
  assert dispatched == [
    btw._Failed(1, 'Side question failed: timed out after 300s')
  ]


class _FakeClient:
  _conversation_id = 'sid'

  def __init__(self) -> None:
    self.sent: list[Any] = []
    self.enqueued: list[Any] = []
    self.closed = False

  def send(self, message: Any) -> None:
    self.sent.append(message)

  def enqueue(self, message: Any) -> None:
    self.enqueued.append(message)


def _stream_event(text: str) -> Any:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      stream_content=agent_pb2.StreamContent(content=text)
    )
  )


def _assistant_event(text: str) -> Any:
  chat = agent_pb2.ChatRecord(success=True)
  chat.assistant.content = text
  record = agent_pb2.AgentMessage(id='a', chat=chat)
  record.inference.SetInParent()
  return agent_pb2.SessionMessage(agent_message=record)


def _completed_event() -> Any:
  delta = agent_pb2.RuntimeDelta()
  delta.turn_completed.SetInParent()
  return agent_pb2.SessionMessage(runtime_delta=delta)


def _hitl_event() -> Any:
  request = agent_pb2.HITLRequest()
  request.call.id = 'call-1'
  return agent_pb2.SessionMessage(hitl_request=request)


def _session(conversation_id: str) -> Any:
  return types.SimpleNamespace(
    session=types.SimpleNamespace(conversation_id=conversation_id)
  )


def _fake_context(client: _FakeClient, script: list[Any]) -> Any:
  @contextlib.asynccontextmanager
  async def create_agent(options: Any, on_event: Any, on_error: Any) -> Any:
    for event in script:
      on_event(event)
    try:
      yield client
    finally:
      client.closed = True

  return types.SimpleNamespace(
    workspace='w',
    create_agent=create_agent,
    transcript=list,
    get_state=lambda: _session('conv'),
  )


def _ext(dispatched: list[Any]) -> Any:
  ext = btw.Btw()
  ext.dispatch = lambda context, action: dispatched.append(action)  # type: ignore[method-assign]
  return ext


def test_fake_client_streams_chunks_then_final() -> None:
  dispatched: list[Any] = []
  client = _FakeClient()
  script = [
    _stream_event('he'),
    _stream_event('llo'),
    _assistant_event('hello'),
    _completed_event(),
  ]
  context = _fake_context(client, script)
  asyncio.run(_ext(dispatched).handle_effect(context, btw._Run(1, 'q')))
  assert client.sent[0].set_approval_mode.mode == agent_pb2.APPROVAL_MODE_HITL
  assert client.enqueued[0].chat.user.content == 'q'
  assert dispatched == [
    btw._Chunk(1, 'he'),
    btw._Chunk(1, 'llo'),
    btw._Final(1, 'hello'),
  ]
  assert client.closed


def test_hitl_request_is_denied() -> None:
  dispatched: list[Any] = []
  client = _FakeClient()
  script = [
    _hitl_event(),
    _assistant_event('ok'),
    _completed_event(),
  ]
  context = _fake_context(client, script)
  asyncio.run(_ext(dispatched).handle_effect(context, btw._Run(1, 'q')))
  responses = [
    m.hitl_response for m in client.sent if m.HasField('hitl_response')
  ]
  assert len(responses) == 1
  assert responses[0].call_id == 'call-1' and responses[0].approved is False


def test_cancel_stops_task_and_clears_registry() -> None:
  async def scenario() -> None:
    dispatched: list[Any] = []
    client = _FakeClient()
    context = _fake_context(client, [])  # No events: _stream waits.
    ext = _ext(dispatched)
    running = asyncio.create_task(ext.handle_effect(context, btw._Run(1, 'q')))
    for _ in range(5):
      await asyncio.sleep(0)
    assert 'w' in ext._tasks
    await ext.handle_effect(context, 'cancel')
    await running
    assert ext._tasks == {}
    assert client.closed
    assert dispatched == []

  asyncio.run(scenario())


def test_on_unload_cancels_running_task() -> None:
  async def scenario() -> None:
    dispatched: list[Any] = []
    client = _FakeClient()
    context = _fake_context(client, [])
    ext = _ext(dispatched)
    running = asyncio.create_task(ext.handle_effect(context, btw._Run(1, 'q')))
    for _ in range(5):
      await asyncio.sleep(0)
    task = ext._tasks['w']
    await ext.on_unload(None)
    await running
    assert task.cancelled() and ext._tasks == {}
    assert client.closed

  asyncio.run(scenario())


def test_caller_cancellation_is_not_swallowed() -> None:
  async def scenario() -> None:
    client = _FakeClient()
    context = _fake_context(client, [])
    ext = _ext([])
    running = asyncio.create_task(ext.handle_effect(context, btw._Run(1, 'q')))
    for _ in range(5):
      await asyncio.sleep(0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
      await running
    assert ext._tasks == {} and client.closed

  asyncio.run(scenario())


def test_output_stops_when_conversation_changes() -> None:
  dispatched: list[Any] = []
  client = _FakeClient()
  script = [_stream_event('he'), _assistant_event('he'), _completed_event()]
  context = _fake_context(client, script)
  ids = iter(['first'] + ['second'] * 50)  # The session changes after start.
  context.get_state = lambda: _session(next(ids))
  asyncio.run(_ext(dispatched).handle_effect(context, btw._Run(1, 'q')))
  assert dispatched == []


@pytest.fixture
def stock_core(monkeypatch: Any) -> None:
  """Behave as on a Mirror that has no panel region."""
  monkeypatch.setattr(btw, '_PANEL_SUPPORTED', False)
  for name in ('_ClosePanel', '_ScrollPanel', '_CopyPanel'):
    monkeypatch.setattr(btw, name, btw._Unsupported)


def test_fallback_uses_widget_region(stock_core: None) -> None:
  from mirror.cli import extension

  region = btw.Btw().render_region()
  assert region is extension.RenderRegion.WIDGET_ABOVE_PROMPT


def test_fallback_render_shows_title_answer_and_dismiss_hint(
  stock_core: None,
) -> None:
  ext = btw.Btw()
  state = btw.BtwState('done', run=1, question='q', answer='a\nb')
  texts = _texts(ext.render(_global(state), state))
  assert texts[0].strip() == '/btw q'
  assert 'a\nb' in texts
  assert texts[-1].strip() == btw._WIDGET_HINT
  assert 'esc' not in texts[-1]


def test_fallback_render_shows_more_rows_than_panel(stock_core: None) -> None:
  ext = btw.Btw()
  answer = '\n'.join(f'line{i}' for i in range(15))
  state = btw.BtwState('done', run=1, question='q', answer=answer)
  body = _texts(ext.render(_global(state), state))[1]
  assert body.splitlines() == [f'line{i}' for i in range(15)]


@needs_panel
def test_fallback_ignores_panel_actions(stock_core: None) -> None:
  state = btw.BtwState('done', run=1, question='q', answer='x\n' * 40)
  for action in (
    actions.ClosePanel(),
    actions.ScrollPanel(3),
    actions.CopyPanel(),
  ):
    same, emitted = _core(state, action)
    assert same == state and emitted == ()


def test_fallback_command_without_question_still_dismisses(
  stock_core: None,
) -> None:
  state = btw.BtwState('done', run=1, question='q', answer='x')
  cleared, emitted = _reduce(state, btw._Ask(''))
  assert cleared.phase == 'idle'
  assert [e.payload for e in emitted] == ['cancel']


def test_extension_id_matches_repository_name() -> None:
  assert btw.Btw.id() == 'mirror-btw-extension'
