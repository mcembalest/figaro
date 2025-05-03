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
        """Test Cmaj jazz ii-V-i progression."""
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
        """Test E blues scale riff."""
        riff = [64, 67, 69, 70, 71, 72] * 2 + [64]
        result = self.detector.analyze(riff)
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], Key.E)
        self.assertGreater(result['confidence'], WEAK_CONFIDENCE_THRESHOLD)

    def test_simple_c_major(self):
        """Test a simple C major scale, expecting high confidence."""
        scale = [60, 62, 64, 65, 67, 69, 71, 72] * 2 # Two octaves
        result = self.detector.analyze(scale)
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], Key.C)
        self.assertEqual(result['mode'], Mode.MAJOR)
        self.assertGreater(result['confidence'], STRONG_CONFIDENCE_THRESHOLD)

    def test_simple_a_minor(self):
        """Test a simple A natural minor scale, expecting high confidence."""
        scale = [69, 71, 72, 74, 76, 77, 79, 81] * 2 # Two octaves starting A4
        result = self.detector.analyze(scale)
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], Key.A)
        self.assertEqual(result['mode'], Mode.MINOR)
        self.assertGreater(result['confidence'], STRONG_CONFIDENCE_THRESHOLD)
        
    def test_ambiguous_cmaj_amin(self):
        """Test notes common to Cmaj/Amin (Am pentatonic), expect lower confidence."""
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
        """Test analyze() with empty note list."""
        self.assertIsNone(self.detector.analyze([]))

    def test_analyze_single_note(self):
        """Test analyze() with single note (insufficient context)."""
        self.assertIsNone(self.detector.analyze([60]))

    def test_validate_notes_invalid_input_type(self):
        """Test _validate_notes raises ValueError for non-list inputs."""
        invalid_inputs = [None, {"a": 60}, np.array([60, 62, 64]), "string"]
        for notes_input in invalid_inputs:
            with self.subTest(notes=notes_input):
                with self.assertRaisesRegex(ValueError, "Input 'notes' must be a list."):
                    self.detector._validate_notes(notes_input)

    def test_validate_notes_empty_list_ok(self):
        """Test _validate_notes handles empty list correctly."""
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
        """Test _validate_notes filters various invalid MIDI note values within a list."""
        valid_notes = self.detector._validate_notes(notes)
        self.assertListEqual(valid_notes, expected_valid_notes, f"Failed for case: {description}")

class TestKrumhanslKeyDetectorPerformance(unittest.TestCase):
    """Test performance and memory usage."""
    
    def setUp(self):
        self.detector = KrumhanslKeyDetector()
        self.performance_threshold = 0.001 # Target ms/note

    def test_realtime_performance(self):
        """Test processing time scales reasonably with input size."""
        for size in [10, 100, 1000, 10000]:
            with self.subTest(size=size):
                notes = np.random.randint(60, 72, size=size).tolist()
                start_time = time.time()
                result = self.detector.analyze(notes)
                elapsed = time.time() - start_time
                self.assertIsNotNone(result)
                self.assertLess(elapsed / size, self.performance_threshold)

    def test_memory_stability(self):
        """Test for memory leaks over repeated calls."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        for _ in range(100):
            notes = np.random.randint(60, 72, size=np.random.randint(100, 1000)).tolist()
            self.detector.analyze(notes)
            gc.collect() # Force GC for accurate test
            
        current_memory = process.memory_info().rss
        memory_increase_mb = (current_memory - initial_memory) / 1024 / 1024
        self.assertLess(memory_increase_mb, 7.0) # Allow 7MB headroom for test overhead