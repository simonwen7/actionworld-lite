"""HTTP contract tests with mocked engine only; never run the real model/simulator.

Requires FastAPI + httpx to be present in the test environment. These are
interface tests, NOT a claim that Linux headless MuJoCo / Vercel is working.
"""
import os
import unittest
from unittest.mock import patch

import numpy as np


class APISyntheticTests(unittest.TestCase):
    def test_mocked_reset_action_and_bad_signature(self):
        from fastapi.testclient import TestClient
        from m8_web.api import app
        from m8_web import engine
        state=np.zeros(51,dtype=np.float64)
        frame=np.zeros((64,64,3),dtype=np.uint8)
        payload={"scored_actions":0,"physics_time_s":.05,"real_rgb_png":"data:image/png;base64,AAAA",
                 "actual":{"actual_xy_m":[0.,0.],"actual_yaw_rad":0.,"pusher_xy_m":[-.2,0.]},
                 "prediction":None,"metrics":None}
        action={"physics_time_s":.10,"real_rgb_png":"data:image/png;base64,AAAA",
                "actual":payload["actual"],"prediction":{"ic01_xy_m":[.01,0.],"m6_xy_m":[.005,0.],
                "correction_m":[-.005,0.],"predicted_contact":False},
                "metrics":{"ic01_next_xy_error_cm":1.0,"m6_next_xy_error_cm":.5}}
        with (patch.dict(os.environ,{"SESSION_SECRET":"m8-interface-only-fake-secret-012345678901234567"}),
              patch.object(engine,"reset_scene",return_value=(payload,state,frame,frame)),
              patch.object(engine,"action_scene",return_value=(action,state,frame,frame)),
              patch.object(engine,"nstate",return_value=51)):
            client=TestClient(app)
            start=client.post('/api/session',json={})
            self.assertEqual(start.status_code,200,start.text)
            token=start.json()['session_token']
            new=client.post('/api/action',json={"session_token":token,"step_size_mm":10,"direction":"right"})
            self.assertEqual(new.status_code,200,new.text)
            self.assertEqual(new.json()['scored_actions'],1)
            bad=client.post('/api/action',json={"session_token":token+'x',"step_size_mm":10,"direction":"right"})
            self.assertEqual(bad.status_code,401)
            invalid=client.post('/api/action',json={"session_token":token,"step_size_mm":99,"direction":"right"})
            self.assertEqual(invalid.status_code,422)


if __name__=='__main__':unittest.main()
