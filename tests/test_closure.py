import json
import unittest
from collections import Counter
from src import paths
from src.closure import load_closure


class ClosureTest(unittest.TestCase):
    def test_reviewed_snapshot_and_release_gate(self):
        p=paths.ROOT/'reports/closure_observability.json'
        snapshot=json.loads(p.read_text())
        health={k:snapshot[k] for k in ('source','dataset_id','mapping_revision','unified_sha256')}
        result=load_closure(health,p)
        self.assertTrue(result['available'])
        self.assertEqual(Counter(r['status'] for r in result['rows']),
                         {'VERIFIABLE':1,'PARTIAL':4,'NOT_OBSERVABLE':16})
        self.assertEqual(len({r['region'] for r in result['rows']}),7)
        for key in ('source','dataset_id','mapping_revision','unified_sha256'):
            bad={**health,key:'different'}
            self.assertFalse(load_closure(bad,p)['available'])


if __name__=='__main__': unittest.main()
