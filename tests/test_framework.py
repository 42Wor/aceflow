import unittest
import numpy as np
import aceflow as af


class TestAceFlowFramework(unittest.TestCase):
    def test_tensor_creation(self):
        t = af.Tensor.zeros([2, 4])
        self.assertEqual(t.shape, [2, 4])
        self.assertTrue(np.allclose(t.numpy(), np.zeros((2, 4))))

    def test_linear_layer(self):
        linear = af.nn.Linear(4, 8)
        x = af.Tensor(np.ones((2, 4), dtype=np.float32))
        out = linear(x)
        self.assertEqual(out.shape, [2, 8])

    def test_rms_norm(self):
        norm = af.nn.RMSNorm(8)
        x = af.Tensor(np.random.randn(2, 8).astype(np.float32))
        out = norm(x)
        self.assertEqual(out.shape, [2, 8])

    def test_multiscale_retention(self):
        ret = af.nn.MultiScaleRetention(d_model=64, num_heads=4)
        x = af.Tensor(np.random.randn(2, 8, 64).astype(np.float32))
        out = ret(x)
        self.assertEqual(out.shape, [2, 8, 64])

    def test_audio_dsp_filters(self):
        filters = af.audio.mel_filters(n_mels=80, n_fft=400, sample_rate=16000)
        self.assertEqual(filters.shape, (80, 201))


if __name__ == "__main__":
    unittest.main()

