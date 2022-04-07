# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import unittest

from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler

import demo
import demo_cqm

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestDemo(unittest.TestCase):

    def test_smoke(self):
        """run demo.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'demo.py')
        subprocess.check_output([sys.executable, demo_file])

    def test_build_dqm(self):
        """Test that DQM built has correct number of variables"""

        W, C, n = demo.read_inputs(flow_file='tests/test_flow.csv', cost_file='tests/test_cost.csv', verbose=False)
        p = 3 
        a = 0.4 

        dqm = demo.build_dqm(W, C, n, p, a, verbose=False)

        self.assertEqual(dqm.num_variables(), n)

    def test_feasible_solns(self):
        """Test that filtered solutions are all feasible"""

        W, C, n = demo.read_inputs(flow_file='tests/test_flow.csv', cost_file='tests/test_cost.csv', verbose=False)
        city_names, _, _ = demo.read_city_info('tests/test_city_data.txt', verbose=False)
        p = 3 
        a = 0.4 

        dqm = demo.build_dqm(W, C, n, p, a, verbose=False)
        sampler = LeapHybridDQMSampler()
        sampleset = sampler.sample_dqm(dqm, label='Test - DQM Airline Hubs')

        ss = list(sampleset.data(['sample']))

        found_valid_soln = False
        for line in ss:
            hubs, legs, valid = demo.get_layout_from_sample(line.sample, city_names, p)

            if valid:

                found_valid_soln = True

                self.assertEqual(len(hubs), p)

                for pair in legs:
                    overlap = set(pair).intersection(hubs)

                    self.assertGreater(len(overlap), 0)

        self.assertTrue(found_valid_soln)

class TestCQMDemo(unittest.TestCase):

    def test_smoke(self):
        """run demo_cqm.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'demo_cqm.py')
        subprocess.check_output([sys.executable, demo_file])

    def test_build_cqm(self):
        """Test that CQM built has correct number of variables"""

        W, C, n = demo_cqm.read_inputs(flow_file='tests/test_flow.csv', cost_file='tests/test_cost.csv', verbose=False)
        p = 3 
        a = 0.4 

        cqm = demo_cqm.build_cqm(W, C, n, p, a, verbose=False)

        self.assertEqual(len(cqm.variables), n**2)

if __name__ == '__main__':
    unittest.main()
