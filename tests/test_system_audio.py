import numpy as np

from audio_analyzer.system_audio import _frames_to_mono, _resolve_soundcard_loopback


def test_frames_to_mono_stereo():
    frames = np.array([[1.0, -1.0], [0.5, 0.5], [-0.25, 0.75]])
    mono = _frames_to_mono(frames)
    np.testing.assert_allclose(mono, [0.0, 0.5, 0.25])


def test_frames_to_mono_already_mono():
    frames = np.array([0.1, -0.2, 0.3])
    mono = _frames_to_mono(frames)
    np.testing.assert_allclose(mono, frames)


def test_frames_to_mono_empty():
    mono = _frames_to_mono(np.empty((0, 2)))
    assert mono.shape == (0,)


class _FakeSpeaker:
    def __init__(self, name: str, id_: str = "speaker-1", channels: int = 2):
        self.name = name
        self.id = id_
        self.channels = channels


class _FakeMic:
    def __init__(self, name: str, isloopback: bool = True):
        self.name = name
        self.isloopback = isloopback


class _FakeSoundCard:
    def __init__(self):
        self.speaker = _FakeSpeaker("Built-in Output")
        self.mic = _FakeMic("Monitor of Built-in Output")

    def default_speaker(self):
        return self.speaker

    def get_speaker(self, name):
        assert name == "Built-in Output"
        return self.speaker

    def get_microphone(self, name, include_loopback=False):
        assert include_loopback is True
        assert name == "Built-in Output"
        return self.mic

    def all_microphones(self, include_loopback=False):
        return [self.mic]


def test_resolve_soundcard_loopback_default_and_named():
    sc = _FakeSoundCard()
    speaker, mic = _resolve_soundcard_loopback(sc, None)
    assert speaker is sc.speaker
    assert mic is sc.mic

    speaker, mic = _resolve_soundcard_loopback(sc, "Built-in Output")
    assert speaker is sc.speaker
    assert mic is sc.mic


class _FakePyAudio:
    def __init__(self):
        self.default = {
            "index": 10,
            "name": "Speakers (Realtek) [Loopback]",
            "maxInputChannels": 2,
            "maxOutputChannels": 0,
            "defaultSampleRate": 48000.0,
            "isLoopbackDevice": True,
        }
        self.alt = {
            "index": 12,
            "name": "Headphones (USB) [Loopback]",
            "maxInputChannels": 2,
            "maxOutputChannels": 0,
            "defaultSampleRate": 48000.0,
            "isLoopbackDevice": True,
        }
        self.output = {
            "index": 2,
            "name": "Speakers (Realtek)",
            "maxInputChannels": 0,
            "maxOutputChannels": 2,
            "defaultSampleRate": 48000.0,
            "isLoopbackDevice": False,
        }

    def get_default_wasapi_loopback(self):
        return self.default

    def get_loopback_device_info_generator(self):
        yield self.default
        yield self.alt

    def get_device_info_by_index(self, index):
        if index == 10:
            return self.default
        if index == 12:
            return self.alt
        if index == 2:
            return self.output
        raise IndexError(index)

    def get_wasapi_loopback_analogue_by_index(self, index):
        assert index == 2
        return self.default


def test_resolve_windows_loopback_default_name_and_output_index():
    from audio_analyzer.system_audio import _resolve_windows_loopback_device

    p = _FakePyAudio()
    assert _resolve_windows_loopback_device(p, None) is p.default
    assert _resolve_windows_loopback_device(p, "Headphones") is p.alt
    assert _resolve_windows_loopback_device(p, "2") is p.default
