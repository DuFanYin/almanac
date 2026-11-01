"""The world's scenarios, from written to scored (model/scenarios.py). At each length one LLM
writes, from today's data and the simulation (the data's world), the scenarios worth telling apart, each in a
direction of its own (a story with the ranges of one or two factors); a simulated path is in every scenario whose
ranges it is in, several or none. Each scenario is read on its own by several readers: how much likelier or less
likely it is than its share of the simulation, for what the simulation does not include (a factor, with what it
misses and why); a reviewer checks their reasons, not their numbers, and the readers of a scenario it found fault
with read it again (data/llm.py scenarios). The probability a scenario should have is the simulation's share times
the median factor; the paths are reweighed so that each has it, the simulation moved least (model/scenarios.py
at_length). What each means for your targets, better or worse than usual, is computed (steps/distribution.py), never
asked.

  write   weekly, every length in one go, revising the scenarios standing; again before the week is up when
          something happens the simulation does not know of yet (a factor's unusual move), at most once in
          EVENT_GAP days. The factors hold until then: they say what the simulation misses, so the simulation's
          daily moves leave them standing
  coach   every COACH_EVERY days once a set is scored: a note on the factors given so far, for every role
  world   daily, each live set on today's simulation: which windows each scenario holds, the windows made for one
          history has too few of, the weights that give each its share times its factor
  score   once a set's length is over and its factors have values (or SCORE_WAIT days after): for each scenario,
          whether the path that happened is in it, against its final probability, the simulation's share and
          history's when written; told to the writer, the readers and the coach, shown on the page. It never changes
          a factor."""

from ...data import llm  # noqa: F401  (the LLM steps, as the step reaches them; the tests stand in for them here)
from .context import record_text, uncovered  # noqa: F401
from .page import explained, for_page, notes_for_page, readings_now  # noqa: F401
from .score import scored  # noqa: F401
from .sets import LENGTHS, lengths, live  # noqa: F401
from .world import for_world  # noqa: F401
from .write import changes, events, write  # noqa: F401
