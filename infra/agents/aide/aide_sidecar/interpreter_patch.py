"""Preserve AIDE's per-node working_dir for post-hoc state predicates.

AIDE's Interpreter.run() writes runfile.py, executes it, then deletes
working_dir (aide/interpreter.py:142-164). For Layer-2b state predicates
(submission_valid, checkpoint_saved, etc.) we need the post-exec artifacts.

Implementation deferred to task #63 — placeholder kept so __init__.py
imports stay stable across iterations.
"""

# TODO(task #63): wrap aide.interpreter.Interpreter.run to snapshot
# working_dir to $MLEVAL_OUTPUT_DIR/working_dirs/node_<step>/ before the
# native cleanup runs.
