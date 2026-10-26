"""Run: python -B -m unittest tests.test_m8_synthetic -v

Pure synthetic protocol tests, no simulator, no checkpoint, no dataset, no TEST.
"""
import hashlib
import time
import unittest
import numpy as np

from m8_web import session, model_adapter


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.secret = "m8-synthetic-not-real-secret-01234567890123456789"
        self.state = np.linspace(-.2,.4,51,dtype=np.float64)
        self.p = np.random.default_rng(7).integers(0,256,(64,64,3),dtype=np.uint8)
        self.c = np.random.default_rng(8).integers(0,256,(64,64,3),dtype=np.uint8)
        self.fp = "a"*64
        self.now = 1_800_000_000

    def test_signed_state_roundtrip_and_renewal(self):
        data=session.new_payload(self.state,self.p,self.c,self.fp,now=self.now)
        token=session.sign(data,self.secret)
        self.assertLess(len(token),session.TOKEN_LIMIT)
        recovered=session.verify(token,self.secret,now=self.now+2)
        s,p,c=session.extract(recovered,self.fp,51)
        np.testing.assert_array_equal(s,self.state)
        np.testing.assert_array_equal(p,self.p)
        np.testing.assert_array_equal(c,self.c)
        updated=session.next_payload(recovered,s,p,c)
        self.assertEqual(updated['steps'],1)
        self.assertEqual(updated['sid'],data['sid'])
        self.assertEqual(updated['exp'],data['exp'])

    def test_tampered_and_expired_tokens(self):
        data=session.new_payload(self.state,self.p,self.c,self.fp,now=self.now)
        token=session.sign(data,self.secret)
        broken=('A' if token[0]!='A' else 'B')+token[1:]
        with self.assertRaises(session.SessionError):session.verify(broken,self.secret,now=self.now)
        with self.assertRaises(session.SessionError):session.verify(token,self.secret,now=self.now+session.LIFETIME_SECONDS)
        with self.assertRaises(session.SessionError):session.extract(data,"b"*64,51)
        with self.assertRaises(session.SessionError):session.extract(data,self.fp,52)

    def test_replay_is_possible_and_explicit(self):
        token=session.sign(session.new_payload(self.state,self.p,self.c,self.fp,now=self.now),self.secret)
        a=session.verify(token,self.secret,now=self.now)
        b=session.verify(token,self.secret,now=self.now)
        self.assertEqual(a,b) # No database: cannot claim at-most-once consumption.

    def test_bad_frame_and_nonfinite_rejected(self):
        with self.assertRaises(session.SessionError):session.new_payload(np.array([float('nan')]),self.p,self.c,self.fp,now=self.now)
        with self.assertRaises(session.SessionError):session.new_payload(self.state,self.p[:32],self.c,self.fp,now=self.now)

    def test_64_action_ceiling(self):
        d=session.new_payload(self.state,self.p,self.c,self.fp,now=self.now)
        for _ in range(session.MAX_STEPS):d=session.next_payload(d,self.state,self.p,self.c)
        with self.assertRaises(session.SessionError):session.next_payload(d,self.state,self.p,self.c)

    def test_metrics_physical_units(self):
        pred={"ic01_xy_m":[.01,0],"m6_xy_m":[.004,0],"ic01_delta_m":[.01,0],"m6_delta_m":[.004,0],"ic01_yaw_rad":0.,"predicted_contact":True}
        before={"block_xy":[0,0]}
        after={"block_xy":[.005,0],"block_yaw":0.,"pusher_xy":[-.19,0],"pusher_contact":True}
        m=model_adapter.score_step(pred,before,after)
        self.assertAlmostEqual(m['ic01_next_xy_error_cm'],.5)
        self.assertAlmostEqual(m['m6_next_xy_error_cm'],.1)
        self.assertTrue(m['contact_correct'])

    def test_real_frame_input_contract(self):
        from m8_web.model_adapter import stride64
        source=np.zeros((256,256,3),dtype=np.uint8)
        small=stride64(source)
        self.assertEqual(small.shape,(64,64,3))
        self.assertEqual(small.dtype,np.uint8)
        with self.assertRaises(ValueError):stride64(np.zeros((64,64,3),dtype=np.uint8))


if __name__=='__main__': unittest.main()
