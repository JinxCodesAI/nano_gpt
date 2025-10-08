import torch
from core.batch import unpack_batch


def test_unpack_batch_language_model_mode():
    b = {
        '_model_mode': 'language_model',
        'x': torch.zeros(2, 4, dtype=torch.long),
        'y': torch.ones(2, 4, dtype=torch.long),
        # extraneous keys shouldn't be used
        'input_ids': torch.full((2, 4), 7, dtype=torch.long),
        'targets': torch.tensor([0.1, 0.2]),
    }
    X, Y = unpack_batch(b)
    assert X.shape == (2, 4)
    assert Y.shape == (2, 4)


def test_unpack_batch_sequence_scorer_mode():
    b = {
        '_model_mode': 'sequence_scorer',
        'x': torch.zeros(2, 4, dtype=torch.long),
        'y': torch.ones(2, 4, dtype=torch.long),
        'input_ids': torch.full((2, 4), 7, dtype=torch.long),
        'targets': torch.tensor([0.1, 0.2]),
    }
    X, Y = unpack_batch(b)
    assert X.shape == (2, 4)
    assert Y.shape == (2,)


def test_unpack_batch_missing_keys_raises():
    b = {
        '_model_mode': 'sequence_scorer',
        'input_ids': torch.full((2, 4), 7, dtype=torch.long),
        # missing targets
    }
    try:
        unpack_batch(b)
        assert False, 'should have raised'
    except KeyError:
        pass

