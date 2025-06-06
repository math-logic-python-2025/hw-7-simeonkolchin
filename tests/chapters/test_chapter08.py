# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter08.py

"""Tests all Chapter 8 tasks."""

from tests.predicates.functions_test import *


def test_task1(debug=False):
    test_replace_functions_with_relations_in_model(debug)


def test_task2(debug=False):
    test_replace_relations_with_functions_in_model(debug)


def test_task3(debug=False):
    test_compile_term(debug)


def test_task4(debug=False):
    test_replace_functions_with_relations_in_formula(debug)


def test_task5(debug=False):
    test_replace_functions_with_relations_in_formulas(debug)


def test_task6(debug=False):
    test_replace_equality_with_SAME_in_formulas(debug)


def test_task7(debug=False):
    test_add_SAME_as_equality_in_model(debug)


def test_task8(debug=False):
    test_make_equality_as_SAME_in_model(debug)


test_task1(True)
test_task2(True)
test_task3(True)
test_task4(True)
test_task5(True)
test_task6(True)
test_task7(True)
test_task8(True)
