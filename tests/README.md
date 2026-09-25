# Test organization

Test modules are named for the scientific or workflow behavior they protect,
not for when or why they were added. Prefer names such as
`test_observation_time_selection.py` and
`test_modes_of_variability_teleconnections.py`.

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
