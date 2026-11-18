import pytest
from math_verify import parse,verify

from dual_opsd.evaluation import math_answer_markup


@pytest.mark.parametrize('answer,expected',[
    (r'\$32,\!348','32348'),
    (r'\$18.90','18.9'),
    (r'\$36','36'),
    (r'\frac{1}{2}','0.5'),
    (r'$\frac{1}{2}$','0.5'),
])
def test_currency_and_fraction_answers_are_graded_numerically(answer,expected):
    gold=parse(math_answer_markup(answer),fallback_mode='no_fallback')
    prediction=parse(math_answer_markup(expected),fallback_mode='no_fallback')
    assert gold and prediction and verify(gold,prediction)
    assert not verify(gold,parse('$987654321$',fallback_mode='no_fallback'))
