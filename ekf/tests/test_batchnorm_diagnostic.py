"""The diagnostic changes BatchNorm buffers while preserving all learned weights."""

import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

import torch

from evaluation.batchnorm_diagnostic import batchnorm_keys, swap_batchnorm_state, verify_swap
from evaluation.evaluate_eskf import write_json_atomic


class BatchNormDiagnosticTests(unittest.TestCase):
    def test_interrupted_progress_commit_preserves_previous_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.json'
            write_json_atomic(path,{'flights':18})
            with patch.object(Path,'replace',side_effect=OSError('simulated interrupted commit')):
                with self.assertRaises(OSError):write_json_atomic(path,{'flights':19})
            self.assertEqual(json.loads(path.read_text()),{'flights':18})
            write_json_atomic(path,{'flights':19})
            self.assertEqual(json.loads(path.read_text()),{'flights':19})

    def test_swaps_only_running_buffers_including_counters(self):
        network = torch.nn.Sequential(torch.nn.Conv1d(3,4,3),torch.nn.BatchNorm1d(4),torch.nn.Linear(4,2))
        original = {key:value.clone() for key,value in network.state_dict().items()}
        donor = {key:value.clone()+7 for key,value in original.items()}
        keys = batchnorm_keys(network)
        self.assertEqual(set(keys),{'1.running_mean','1.running_var','1.num_batches_tracked'})
        for weights,statistics in ((original,original),(donor,donor),(donor,original),(original,donor)):
            result = swap_batchnorm_state(weights,statistics,keys)
            verify_swap(result,weights,statistics,keys)
            self.assertTrue(torch.equal(result['1.weight'],weights['1.weight']))
            self.assertTrue(torch.equal(result['1.bias'],weights['1.bias']))
            for key in keys:self.assertTrue(torch.equal(result[key],statistics[key]))
            result['0.weight'].zero_()
            self.assertFalse(torch.equal(result['0.weight'],weights['0.weight']))

    def test_rejects_shape_mismatch_and_non_buffer_changes(self):
        network = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
        state = network.state_dict();keys = batchnorm_keys(network)
        donor = {key:value.clone() for key,value in state.items()};donor['0.running_var']=torch.ones(4)
        with self.assertRaises(ValueError):swap_batchnorm_state(state,donor,keys)
        result = swap_batchnorm_state(state,state,keys);result['0.weight'].add_(1)
        with self.assertRaisesRegex(ValueError,'Unexpected tensor change'):verify_swap(result,state,state,keys)

    def test_rejects_network_without_batchnorm(self):
        with self.assertRaisesRegex(ValueError,'No BatchNorm'):batchnorm_keys(torch.nn.Linear(3,2))


if __name__=='__main__':unittest.main()
