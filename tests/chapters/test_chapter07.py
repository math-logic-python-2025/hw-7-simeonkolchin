# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter07.py

"""Tests all Chapter 7 tasks."""

from tests.predicates.syntax_test import *
from tests.predicates.semantics_test import *


def test_task1(debug=False):
    test_term_repr(debug)


def test_task2(debug=False):
    test_formula_repr(debug)


def test_task3(debug=False):
    test_term_parse_prefix(debug)
    test_term_parse(debug)


def test_task4(debug=False):
    test_formula_parse_prefix(debug)
    test_formula_parse(debug)


def test_task5(debug=False):
    test_term_constants(debug)
    test_term_variables(debug)
    test_term_functions(debug)


def test_task6(debug=False):
    test_formula_constants(debug)
    test_formula_variables(debug)
    test_free_variables(debug)
    test_formula_functions(debug)
    test_relations(debug)


def test_task7(debug=False):
    test_evaluate_term(debug)


def test_task8(debug=False):
    test_evaluate_formula(debug)


def test_task9(debug=False):
    test_is_model_of(debug)


test_task1(True)
test_task2(True)
test_task3(True)
test_task4(True)
test_task5(True)
test_task6(True)
test_task7(True)
test_task8(True)
test_task9(True)
