import numpy as np

import deep_dive


def test_detected_keeps_the_specificity():
    labels = np.array([0] * 80 + [1] * 5)
    scores = np.r_[np.arange(80.0), [100, 79.5, 78.5, 70, -1]]
    hit = deep_dive.detected(scores, labels)
    assert hit[labels == 0].sum() == 4                      # exactly 5% of 80 normal slides get through
    assert hit[80:].tolist() == [True, True, True, False, False]    # threshold is the 5th-highest normal (75)


def test_size_bins_and_centres():
    assert [deep_dive.size_bin(n) for n in (1, 10, 11, 200, 201, 5000)] == ["1-10", "1-10", "11-50", "51-200", ">200", ">200"]
    assert deep_dive.centre_of(0.243094) == "RUMC" and deep_dive.centre_of(0.226907) == "UMCU"
