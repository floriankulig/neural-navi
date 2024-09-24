import numpy as np


def normalize(response, input_range, output_range):
    value = (
        np.interp(response.value.magnitude, input_range, output_range)
        if response is not None and response.value is not None
        else None
    )
    return round2(value)


def numeric_or_none(response):
    value = (
        response.value.magnitude
        if response is not None and response.value is not None
        else None
    )
    return round2(value)


def round2(value):
    return round(value, 2) if value is not None else None
