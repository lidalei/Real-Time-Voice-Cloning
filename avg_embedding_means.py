import argparse
from pathlib import Path

import typing

import numpy as np
import scipy.spatial.distance

from encoder.inference import Model as EncoderModel
from synthesizer.inference import Synthesizer

_NUM_ENROLLMENTS = 3
_NUM_VERIFICATIONS = 5
_WAV_FODLER = Path('/Users/dalei/Downloads/VCTK-Corpus/wav48')
_TXT_FODLER = Path('/Users/dalei/Downloads/VCTK-Corpus/txt')


def run(args: argparse.Namespace):
    # Load encoder model
    encoder = EncoderModel()
    encoder.load(Path('encoder/saved_models/pretrained.pt'))

    # [p304, p305, ...]
    speaker_dirs = [f.parts[-1] for f in _WAV_FODLER.glob("*") if f.is_dir()]
    if len(speaker_dirs) == 0:
        raise Exception("No speakers found. Make sure you are pointing to the directory")

    # 'p304' -> [001.wav, 002.wav, ...]
    speaker_utterances = dict()  # type: typing.Dict[str, typing.List[str]]
    for d in speaker_dirs:
        speaker_utterances[d] = [w.parts[-1] for w in _WAV_FODLER.joinpath(d).glob('*.wav')]

    speaker_embeddings = dict()  # type: typing.Dict[str, np.ndarray]
    no_use_speaker_embeddings = dict()  # type: typing.Dict[str, np.ndarray]
    for d in speaker_utterances:
        utterances = speaker_utterances[d]
        enrollments = utterances[:_NUM_ENROLLMENTS]
        print(f'speaker: {d}, enrollments: {enrollments}')
        audios = [_WAV_FODLER.joinpath(d, u) for u in enrollments]
        speaker_embeddings[d] = encoder.embed_speaker(audios, using_partials=True)
        no_use_speaker_embeddings[d] = encoder.embed_speaker(audios, using_partials=False)

    # Different speaker
    for d in speaker_utterances:
        utterances = speaker_utterances[d]
        # Repeat 5 times
        for utterance in np.random.choice(utterances, size=_NUM_VERIFICATIONS, replace=False):  # type: str
            txt = _TXT_FODLER.joinpath(d, utterance).with_suffix('.txt')
            text = txt.read_text()

            # using partials
            utterance_embedding = encoder.embed_utterance(
                _WAV_FODLER.joinpath(d, utterance),
                source_sr=Synthesizer.sample_rate,
                using_partials=True,
            )
            cosine_similarity = 1.0 - scipy.spatial.distance.cosine(speaker_embeddings[d], utterance_embedding)
            print(f'use: speaker: {d}, utterance: {utterance}, text: {text}, sim: {cosine_similarity}')

            # not using partials
            utterance_embedding = encoder.embed_utterance(
                _WAV_FODLER.joinpath(d, utterance),
                source_sr=Synthesizer.sample_rate,
                using_partials=False,
            )
            cosine_similarity = 1.0 - scipy.spatial.distance.cosine(no_use_speaker_embeddings[d], utterance_embedding)
            print(f'no_use: speaker: {d}, utterance: {utterance}, text: {text}, sim: {cosine_similarity}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    run(args)

