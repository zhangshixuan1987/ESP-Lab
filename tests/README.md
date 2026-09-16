# Test organization

Test modules are named for the scientific or workflow behavior they protect,
not for when or why they were added. Prefer concise, readable domain prefixes such as
`test_obs_time_selection.py`, `test_mov_teleconnections.py`, and `test_eli_diagnostics.py`.

Maintenance rules:

- keep one authoritative module for each behavior domain;
- do not add numbered, `new`, `orig`, or implementation-experiment suffixes;
- test production functions rather than copying their logic into a test;
- use deterministic synthetic data and explicit expected coordinates or values;
- express scientific invariants, such as conservation, equivalent longitude
  conventions, valid probability bounds, or preservation of a uniform field;
- skip unavailable optional runtimes explicitly, without weakening assertions
  when the runtime is available;
- place notebook architecture and resource-lifecycle checks in descriptively
  named notebook test modules.
