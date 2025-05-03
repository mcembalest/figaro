import unittest
import numpy as np
import time
import gc
import psutil
import os
from enum import IntEnum, Enum
from figaro import KrumhanslKeyDetector
import logging
from parameterized import parameterized

# Configure basic logging for the test module to ensure assertLogs works reliably
logging.basicConfig(level=logging.DEBUG)

class Key(IntEnum):
    """MIDI pitch classes (C=0..B=11)."""
    C = 0
    C_SHARP = 1
    D = 2
    D_SHARP = 3
    E = 4
    F = 5
    F_SHARP = 6
    G = 7
    G_SHARP = 8
    A = 9
    A_SHARP = 10
    B = 11

class Mode(str, Enum):
    """Musical modes."""
    MAJOR = "major"
    MINOR = "minor"

# Empirically derived confidence thresholds
STRONG_CONFIDENCE_THRESHOLD = 0.80
MODERATE_CONFIDENCE_THRESHOLD = 0.65
WEAK_CONFIDENCE_THRESHOLD = 0.45

class TestKrumhanslKeyDetectorRealWorld(unittest.TestCase):
    """Test with common musical patterns."""
    
    def setUp(self):
        self.detector = KrumhanslKeyDetector()

    def test_jazz_ii_v_i(self):
        """
        Importance: Medium.
        Quality: Okay-ish, but Isolated.

        Testing a ii-V-i? Cute. At least it's a real musical pattern.
        But this is *perfect* MIDI data. What about mistuned notes from Yin?
        What about extra noise notes? What about timing variations?
        This tests the algorithm in a vacuum, not its robustness in a real-time Pyo environment.
        """
        progression = [
            60, 64, 67, 71, 60, 64, 67, 71, # Cmaj7
            62, 65, 69, 72,               # Dm7
            66, 67, 70, 73,               # Chromatic approach
            67, 71, 74, 77,               # G7
            72, 76, 79, 83                # Cmaj7 (high)
        ]
        result = self.detector.analyze(progression)
        self.assertIsNotNone(result)
        self.assertIn(result['key'], [Key.C, Key.G])
        self.assertEqual(result['mode'], Mode.MAJOR)
        self.assertGreater(result['confidence'], MODERATE_CONFIDENCE_THRESHOLD)

    def test_blues_riff(self):
        """
        Importance: Medium.
        Quality: Okay-ish, Still Isolated.

        A blues riff? Slightly more realistic with the bent notes implied.
        But again, *perfect* MIDI input. The real challenge is detecting key from the messy,
        often ambiguous output of a real-time pitch tracker like Yin, especially with slides
        or vibrato. This test ignores all that messy reality. Confidence threshold check is arbitrary.
        """
        riff = [64, 67, 69, 70, 71, 72] * 2 + [64]
        result = self.detector.analyze(riff)
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], Key.E)
        self.assertGreater(result['confidence'], WEAK_CONFIDENCE_THRESHOLD)

    def test_simple_c_major(self):
        """
        Importance: Low.
        Quality: Trivial.

        Testing a C major scale? If your key detector can't get this right,
        just delete the whole project. This is the absolute baseline.
        The high confidence expectation is meaningless without comparing it to
        performance on *difficult* inputs. Provides almost zero useful information.
        """
        scale = [60, 62, 64, 65, 67, 69, 71, 72] * 2 # Two octaves
        result = self.detector.analyze(scale)
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], Key.C)
        self.assertEqual(result['mode'], Mode.MAJOR)
        self.assertGreater(result['confidence'], STRONG_CONFIDENCE_THRESHOLD)

    def test_simple_a_minor(self):
        """
        Importance: Low.
        Quality: Trivial.

        Same as the C major test, just for minor. Equally pointless for proving
        robustness or real-world applicability. If it fails this, it's fundamentally broken,
        but passing tells you nothing interesting.
        """
        scale = [69, 71, 72, 74, 76, 77, 79, 81] * 2 # Two octaves starting A4
        result = self.detector.analyze(scale)
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], Key.A)
        self.assertEqual(result['mode'], Mode.MINOR)
        self.assertGreater(result['confidence'], STRONG_CONFIDENCE_THRESHOLD)
        
    def test_ambiguous_cmaj_amin(self):
        """
        Importance: Medium.
        Quality: Slightly Better, Still Isolated.

        Testing an ambiguous pentatonic scale is a *slightly* better idea.
        Checking for reduced confidence makes sense. But STILL using perfect MIDI.
        How does it handle ambiguity when the input notes themselves are fluctuating
        or noisy from Yin? That's the *real* test. This is just theory.
        """
        # A C D E G - common notes, lacks B (Cmaj) and F# (Gmaj)/F (Cmaj/Amin) differentiation
        notes = [69, 72, 74, 76, 79] * 4 # A C D E G repeated
        result = self.detector.analyze(notes)
        self.assertIsNotNone(result)
        # It might guess C major or A minor, we care more about reduced confidence
        self.assertIn(result['key'], [Key.C, Key.A])
        # Confidence should be penalized due to ambiguity
        self.assertLess(result['confidence'], STRONG_CONFIDENCE_THRESHOLD)
        # Possibly even below moderate, but let's be safe
        self.assertGreater(result['confidence'], WEAK_CONFIDENCE_THRESHOLD * 0.8) # Ensure it's not zero

class TestKrumhanslKeyDetectorEdgeCases(unittest.TestCase):
    """Test edge cases and invalid inputs."""
    
    def setUp(self):
        self.detector = KrumhanslKeyDetector()

    def test_analyze_empty_input(self):
        """
        Importance: Microscopic.
        Quality: Defensive Programming 101.

        Tests if it handles an empty list. Okay, prevents one crash.
        Totally unrelated to musical key detection or Pyo.
        """
        self.assertIsNone(self.detector.analyze([]))

    def test_analyze_single_note(self):
        """
        Importance: Microscopic.
        Quality: Obvious Constraint Check.

        Tests if it correctly requires a minimum number of notes.
        Again, basic input validation, not testing the core logic's quality.
        """
        self.assertIsNone(self.detector.analyze([60]))

    def test_validate_notes_invalid_input_type(self):
        """
        Importance: Negligible.
        Quality: Testing Python?!

        Checks if it raises an error when you pass garbage instead of a list.
        This is testing basic Python type handling. Utterly pointless.
        If your calling code passes garbage, that's the caller's bug.
        """
        invalid_inputs = [None, {"a": 60}, np.array([60, 62, 64]), "string"]
        for notes_input in invalid_inputs:
            with self.subTest(notes=notes_input):
                with self.assertRaisesRegex(ValueError, "Input 'notes' must be a list."):
                    self.detector._validate_notes(notes_input)

    def test_validate_notes_empty_list_ok(self):
        """
        Importance: Negligible.
        Quality: More Python Type Trivia.

        Checks if it *doesn't* crash on an empty list. Combines nicely
        with the *other* test that checks it returns None for an empty list.
        Still has nothing to do with audio or key detection quality.
        """
        try:
            result = self.detector._validate_notes([])
            self.assertEqual(result, [])
        except ValueError:
            self.fail("_validate_notes raised ValueError on empty list")

    @parameterized.expand([
        ([-1, 60, 64], [60, 64], "Negative MIDI note"),
        ([60, 128, 64], [60, 64], "MIDI note > 127"),
        ([60, None, 64], [60, 64], "None value"),
        ([60, 'invalid', 64], [60, 64], "String value"),
        ([60, 64.5, 67], [60, 67], "Float non-integer MIDI"),
        # Special case: float that IS an integer should be allowed
        ([60, 64.0, 67], [60, 64, 67], "Float integer MIDI"),
        ([60, np.int64(64), 67], [60, 64, 67], "Numpy integer"),
        ([60, np.float64(64.0), 67], [60, 64, 67], "Numpy float integer"),
    ])
    def test_validate_notes_filters_invalid_elements(self, notes, expected_valid_notes, description):
        """
        Importance: Low.
        Quality: Pedantic, but Thoroughly Irrelevant.

        Tests filtering various kinds of bad data *within* the list.
        It's good that it handles this stuff defensively, but this is far removed
        from the core problem. Does this help Figaro jam better? No.
        It just proves your input sanitizer works on hypothetical garbage.
        """
        valid_notes = self.detector._validate_notes(notes)
        self.assertListEqual(valid_notes, expected_valid_notes, f"Failed for case: {description}")

class TestKrumhanslKeyDetectorPerformance(unittest.TestCase):
    """Test performance and memory usage."""
    
    def setUp(self):
        self.detector = KrumhanslKeyDetector()
        self.performance_threshold = 0.001 # Target ms/note

    def test_realtime_performance(self):
        """
        Importance: Medium.
        Quality: Naive.

        Checks processing time scales linearly with random MIDI notes.
        Okay, it's fast on *average* for simple data. But real-time isn't about average,
        it's about *worst-case* latency. Does it ever spike unpredictably?
        Does performance degrade with *musically complex* input vs random notes?
        This test doesn't know. It's a superficial benchmark.
        """
        for size in [10, 100, 1000, 10000]:
            with self.subTest(size=size):
                notes = np.random.randint(60, 72, size=size).tolist()
                start_time = time.time()
                result = self.detector.analyze(notes)
                elapsed = time.time() - start_time
                self.assertIsNotNone(result)
                self.assertLess(elapsed / size, self.performance_threshold)

    def test_memory_stability(self):
        """
        Importance: Medium.
        Quality: Okay, but Coarse.

        Checks for gross memory leaks over many calls. Good hygiene.
        But the check is very coarse (7MB threshold?). It might miss slow leaks.
        More importantly, it doesn't test memory usage *during* Pyo's audio callback.
        Does running this *alongside* Pyo cause memory pressure or GC pauses that kill real-time performance?
        This isolated test can't tell you.
        """
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        for _ in range(100):
            notes = np.random.randint(60, 72, size=np.random.randint(100, 1000)).tolist()
            self.detector.analyze(notes)
            gc.collect() # Force GC for accurate test
            
        current_memory = process.memory_info().rss
        memory_increase_mb = (current_memory - initial_memory) / 1024 / 1024
        self.assertLess(memory_increase_mb, 7.0) # Allow 7MB headroom for test overhead