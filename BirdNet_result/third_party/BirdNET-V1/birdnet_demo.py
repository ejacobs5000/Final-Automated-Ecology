"""CPU inference adapter for the original BirdNET-V1 snapshot.

Architecture adapted from BirdNET-V1/model/model.py (CC BY-NC-SA 4.0).
Original authors: Stefan Kahl and the BirdNET team.
This adapter is demo code, not an official BirdNET implementation.
"""
from pathlib import Path
import collections
import copyreg
import hashlib
import pickle
import numpy as np

MODEL_SHA256 = '979c6133147354a87f8ee62cc0d77182259014e49aa0c5eb04d99614729a839e'

class LegacyMetadata:
    """Inert placeholder for the unused serialized Theano/Lasagne graph."""
    def __init__(self, *args, **kwargs):
        self.args = args

class SnapshotReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('lasagne.', 'theano.')) or (module, name) == ('model.lasagne_net', 'logmeanexp'):
            return type(name, (LegacyMetadata,), {})
        allowed = {
            ('copy_reg', '_reconstructor'): copyreg._reconstructor,
            ('__builtin__', 'object'): object,
            ('__builtin__', 'set'): set,
            ('collections', 'OrderedDict'): collections.OrderedDict,
        }
        if (module, name) in allowed:
            return allowed[module, name]
        if module == 'numpy' and name in ('ndarray', 'dtype', 'float64', 'float32', 'int64', 'int32'):
            return getattr(np, name)
        if (module, name) in [('numpy.core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar')]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Unsupported snapshot object: {module}.{name}')

def load_snapshot(path):
    path = Path(path)
    if MODEL_SHA256:
        with path.open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError('Model digest differs from the recorded download.')
    with path.open('rb') as f:
        snapshot = SnapshotReader(f, encoding='latin1').load()
    return snapshot

def graph_layers(output):
    ordered, seen = [], set()
    def visit(node):
        if id(node) in seen:
            return
        parents = getattr(node, 'input_layers', [getattr(node, 'input_layer', None)])
        for parent in parents:
            if parent is not None:
                visit(parent)
        seen.add(id(node))
        ordered.append(node)
    visit(output)
    return ordered

def shared_array(variable):
    value = variable.container.storage[0]
    if isinstance(value, np.ndarray):
        return value
    if type(value).__name__ == 'GpuArray_unpickler' and isinstance(value.args[0], np.ndarray):
        return value.args[0]
    raise ValueError(f'Unexpected tensor storage: {type(value)}')

import torch
from torch import nn
from torch.nn import functional as F

class BirdNETV1(nn.Module):
    """Evaluate the downloaded Lasagne graph with PyTorch CPU operations.

    Stored batch-normalization inverse standard deviations are used directly.
    Kernels are flipped when the saved Lasagne convolution requests true
    convolution (PyTorch uses cross-correlation). Dropout is disabled.
    """
    def __init__(self, snapshot):
        super().__init__()
        self.layers = graph_layers(snapshot['net'])
        self.indices = {id(node): i for i, node in enumerate(self.layers)}
        self.classes = list(snapshot['classes'])
        self.trainable_count = 0
        self.stored_count = 0
        seen = set()
        for i, node in enumerate(self.layers):
            for variable, tags in node.params.items():
                if id(variable) not in seen:
                    a = shared_array(variable)
                    self.stored_count += a.size
                    self.trainable_count += a.size if 'trainable' in tags else 0
                    seen.add(id(variable))
            for key in ('W', 'b', 'beta', 'gamma', 'mean', 'inv_std'):
                variable = getattr(node, key, None)
                if variable is not None:
                    a = shared_array(variable)
                    if key == 'W' and node.flip_filters:
                        a = a[..., ::-1, ::-1]
                    self.register_buffer(f'n{i}_{key}', torch.from_numpy(np.array(a, dtype=np.float32, copy=True)))
        self.eval()

    def forward(self, x):
        if tuple(x.shape[1:]) != (1, 64, 384):
            raise ValueError(f'Expected (batch, 1, 64, 384), got {tuple(x.shape)}')
        values = []
        for i, node in enumerate(self.layers):
            kind = type(node).__name__
            get = lambda key: getattr(self, f'n{i}_{key}', None)
            if kind == 'InputLayer':
                y = x
            elif kind == 'ElemwiseSumLayer':
                y = sum(c * values[self.indices[id(p)]] for c, p in zip(node.coeffs, node.input_layers))
            else:
                a = values[self.indices[id(node.input_layer)]]
                if kind == 'Conv2DLayer':
                    assert node.nonlinearity.__name__ == 'linear'
                    pad = tuple(k // 2 for k in node.filter_size) if node.pad == 'same' else node.pad
                    y = F.conv2d(a, get('W'), get('b'), stride=node.stride, padding=pad, groups=node.num_groups)
                elif kind in ('BatchNormDNNLayer', 'BatchNormLayer'):
                    assert node.axes == (0, 2, 3)
                    beta, gamma, mean, inv = [get(k)[None, :, None, None] for k in ('beta', 'gamma', 'mean', 'inv_std')]
                    y = (a - mean) * (gamma * inv) + beta
                elif kind == 'NonlinearityLayer':
                    activation = node.nonlinearity.__name__
                    if activation == 'rectify': y = F.relu(a)
                    elif activation == 'sigmoid': y = torch.sigmoid(a)
                    elif activation == 'linear': y = a
                    else: raise ValueError(activation)
                elif kind in ('MaxPool2DLayer', 'Pool2DLayer'):
                    assert node.ignore_border
                    if node.mode == 'max':
                        y = F.max_pool2d(a, node.pool_size, node.stride, node.pad)
                    elif node.mode == 'average_exc_pad':
                        y = F.avg_pool2d(a, node.pool_size, node.stride, node.pad, count_include_pad=False)
                    else: raise ValueError(node.mode)
                elif kind == 'DropoutLayer':
                    y = a
                elif kind == 'GlobalPoolLayer':
                    assert node.pool_function.__name__ == 'logmeanexp'
                    flattened = a.flatten(2)
                    y = (torch.logsumexp(5 * flattened, dim=2) - np.log(flattened.shape[2])) / 5
                else:
                    raise ValueError(f'Unsupported layer: {kind}')
            values.append(y)
        return values[-1]

from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
from audio_preprocessing import spectrogram

def read_audio(path, target_rate=48000):
    rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / float(2 ** (np.iinfo(audio.dtype).bits - 1))
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != target_rate:
        divisor = gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, rate // divisor).astype(np.float32)
    return audio, target_rate

def make_spectrogram(segment, snapshot):
    return spectrogram(segment, snapshot['sample_rate'],
                       shape=(snapshot['im_size'][1], snapshot['im_size'][0]),
                       win_len=snapshot['win_len'], fmin=snapshot['spec_fmin'],
                       fmax=snapshot['spec_fmax'], magnitude_scale=snapshot['magnitude_scale'],
                       frequency_scale='mel', bandpass=True).astype(np.float32)

def analyze_audio(path, model, snapshot, batch_size=4):
    """Analyze complete 3-second windows; exclude the incomplete final window.

    No geographic/seasonal filtering. Returns start times, all class scores,
    and the audio duration. Scores are model outputs, not calibrated certainty.
    """
    audio, rate = read_audio(path, snapshot['sample_rate'])
    width = int(snapshot['spec_length'] * rate)
    starts = np.arange(0, len(audio) - width + 1, width)
    if len(starts) == 0:
        raise ValueError('Please supply at least three seconds of audio.')
    batches = []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            specs = np.stack([make_spectrogram(audio[t:t + width], snapshot) for t in starts[offset:offset + batch_size]])
            batches.append(model(torch.from_numpy(specs[:, None])).numpy())
    scores = np.concatenate(batches)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Invalid model output.')
    return starts / rate, scores, len(audio) / rate
