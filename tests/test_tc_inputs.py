"""Input orchestration tests that do not launch TempestExtremes."""
import os
import tempfile
import unittest
from pathlib import Path

from workflows.tropical_cyclones.inputs import ensure_tracks, track_paths


class EnsureTracksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/run_process_tc_track_e3sm.py').touch()
        self.kw = dict(repo_root=self.root, track_root=self.root / 'products',
                       cases=['case'], members=['EN00'], parsets=['set3'],
                       settings={'sim_dir': '/raw', 'stream_tag': 'eam.h2'})
        self.calls = []

    def build(self, cmd, **kwargs):
        self.calls.append(cmd)
        case = cmd[cmd.index('--cases') + 1]
        members = cmd[cmd.index('--members') + 1:cmd.index('--parset')]
        parset = cmd[cmd.index('--parset') + 1]
        for member in members:
            track, hist = track_paths(self.kw['track_root'], case, member, parset)
            track.parent.mkdir(parents=True, exist_ok=True)
            old_time = hist.stat().st_mtime_ns if hist.exists() else 0
            track.write_text('')  # Valid zero-storm result must be reusable.
            hist.write_bytes(b'histogram')
            os.utime(hist, ns=(old_time + 1000000000, old_time + 1000000000))

    def test_first_build_then_reuse_without_subprocess(self):
        first = ensure_tracks(**self.kw, runner=self.build)
        second = ensure_tracks(**self.kw, runner=self.build)
        self.assertEqual(first['built'], 1)
        self.assertEqual(second['reused'], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn('-6.0', self.calls[0])
        self.assertIn('--force', self.calls[0])

    def test_require_missing_does_not_write(self):
        with self.assertRaises(FileNotFoundError):
            ensure_tracks(**self.kw, mode='require', runner=self.build)
        self.assertFalse(self.calls)
        self.assertFalse(self.kw['track_root'].exists())

    def test_changed_settings_rebuild_and_require_rejects(self):
        ensure_tracks(**self.kw, runner=self.build)
        self.kw['settings']['min_wind'] = 12
        with self.assertRaises(FileNotFoundError):
            ensure_tracks(**self.kw, mode='require', runner=self.build)
        self.assertEqual(ensure_tracks(**self.kw, runner=self.build)['built'], 1)

    def test_partial_pair_and_failed_processor_are_not_cached(self):
        def incomplete(cmd, **kwargs):
            track, _ = track_paths(self.kw['track_root'], 'case', 'EN00', 'set3')
            track.parent.mkdir(parents=True)
            track.touch()
        with self.assertRaises(RuntimeError):
            ensure_tracks(**self.kw, runner=incomplete)
        self.assertFalse(list(self.root.rglob('*.inputs.json')))

    def test_legacy_pair_is_reported_and_rebuild_is_explicit(self):
        self.build(['--cases', 'case', '--members', 'EN00', '--parset', 'set3'])
        self.calls.clear()
        self.assertEqual(ensure_tracks(**self.kw, mode='require', runner=self.build)['legacy_reused'], 1)
        self.assertFalse(self.calls)
        self.assertEqual(ensure_tracks(**self.kw, mode='rebuild', runner=self.build)['built'], 1)

    def test_members_are_batched_and_unchanged_stale_outputs_rejected(self):
        self.kw['members'] = ['EN00', 'EN01']
        self.assertEqual(ensure_tracks(**self.kw, runner=self.build)['built'], 2)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(RuntimeError, 'unchanged'):
            ensure_tracks(**self.kw, mode='rebuild', runner=lambda *a, **k: None)

    def test_all_requested_methods_are_built(self):
        self.kw['parsets'] = ['set1', 'set2', 'set3', 'set4', 'set5']
        self.assertEqual(ensure_tracks(**self.kw, runner=self.build)['built'], 5)
        last = self.calls[-1]
        self.assertEqual(last[last.index('--wc2') + 1], '')


if __name__ == '__main__':
    unittest.main()
