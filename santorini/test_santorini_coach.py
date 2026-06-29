import os
import pickle
import tempfile
import unittest

from Coach import Coach
from utils import dotdict


class TestSantoriniCoachExamples(unittest.TestCase):
    def make_coach_shell(self, load_folder, load_file='best.pth.tar'):
        coach = object.__new__(Coach)
        coach.args = dotdict({
            'load_folder_file': (load_folder, load_file),
        })
        return coach

    def test_examples_candidates_prefer_explicit_then_resume_files(self):
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, 'checkpoint_1.pth.tar.examples'), 'wb').close()
            open(os.path.join(folder, 'checkpoint_2.pth.tar.examples'), 'wb').close()
            coach = self.make_coach_shell(folder)

            candidates = coach._examplesCandidates('manual.examples')

            self.assertEqual(candidates[0], os.path.join(folder, 'manual.examples'))
            self.assertEqual(candidates[1], 'manual.examples')
            self.assertEqual(candidates[2], os.path.join(folder, 'best.pth.tar.examples'))
            self.assertEqual(candidates[3], os.path.join(folder, 'latest.examples'))
            self.assertEqual(len(candidates), len({os.path.abspath(path) for path in candidates}))

    def test_load_train_examples_falls_back_to_latest(self):
        with tempfile.TemporaryDirectory() as folder:
            examples = [('board', 'policy', 1)]
            with open(os.path.join(folder, 'latest.examples'), 'wb') as examples_file:
                pickle.dump(examples, examples_file)

            coach = self.make_coach_shell(folder)
            coach.loadTrainExamples(skipFirstSelfPlay=False)

            self.assertEqual(coach.trainExamplesHistory, examples)
            self.assertFalse(coach.skipFirstSelfPlay)


if __name__ == "__main__":
    unittest.main()
