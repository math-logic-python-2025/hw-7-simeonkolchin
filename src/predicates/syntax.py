# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: predicates/syntax.py

"""Syntactic handling of predicate-logic expressions."""

from __future__ import annotations
from functools import lru_cache
from typing import AbstractSet, Mapping, Optional, Sequence, Set, Tuple, Union

from src.logic_utils import (
    frozen,
    memoized_parameterless_method,
)

from src.propositions.syntax import (
    Formula as PropositionalFormula,
)


class ForbiddenVariableError(Exception):
    """Raised by `Term.substitute` and `Formula.substitute` when a substituted
    term contains a variable name that is forbidden in that context.

    Attributes:
        variable_name (`str`): the variable name that was forbidden in the
            context in which a term containing it was to be substituted.
    """

    variable_name: str

    def __init__(self, variable_name: str):
        """Initializes a `ForbiddenVariableError` from the offending variable
        name.

        Parameters:
            variable_name: variable name that is forbidden in the context in
                which a term containing it is to be substituted.
        """
        assert is_variable(variable_name)
        self.variable_name = variable_name


@lru_cache(maxsize=100)  # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return (
            ((string[0] >= "0" and string[0] <= "9") or (string[0] >= "a" and string[0] <= "e")) and string.isalnum()
    ) or string == "_"


@lru_cache(maxsize=100)  # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= "u" and string[0] <= "z" and string.isalnum()


@lru_cache(maxsize=100)  # Cache the return value of is_function
def is_function(string: str) -> bool:
    """Checks if the given string is a function name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a function name, ``False`` otherwise.
    """
    return string[0] >= "f" and string[0] <= "t" and string.isalnum()


def separate_bracketed_formula(string: str) -> Tuple:
    if string[0] not in ('[', '('):
        print(f"{string} doesn't start with a bracket")
        return string

    opening = string[0]
    closing = ']' if opening == '[' else ')'

    balance = 0
    for index, char in enumerate(string):
        if char == opening:
            balance += 1
        elif char == closing:
            balance -= 1

        if balance == 0:
            prefix = string[:index + 1]
            suffix = string[index + 1:]
            return prefix, suffix


def separate_term(string: str) -> Tuple:
    assert string, "Input string must be non-empty"
    first_char = string[0]

    if first_char == '_':
        return '_', string[1:]

    idx = 0
    while idx < len(string) and string[idx].isalnum():
        idx += 1

    if is_constant(first_char) or is_variable(first_char):
        return string[:idx], string[idx:]

    if is_function(first_char):
        args_part, remainder = separate_bracketed_formula(string[idx:])
        return string[:idx] + args_part, remainder


def separate_function_or_relation_name(string: str) -> Tuple:
    assert string, "Input string must be non-empty"
    idx = 0

    while idx < len(string) and string[idx].isalnum():
        idx += 1

    name = string[:idx]
    bracketed_args, remainder = separate_bracketed_formula(string[idx:])
    return name, bracketed_args, remainder


def separate_operator(string: str) -> list:
    first_char = string[0]

    if first_char in ('&', '|', '+'):
        return first_char, string[1:]

    if first_char == '-':
        next_char = string[1]
        assert next_char in ('>', '&', '|'), "Invalid operator after '-'"
        return string[:2], string[2:]

    assert string.startswith('<->'), "Expected '<->' operator"
    return string[:3], string[3:]


@frozen
class Term:
    """An immutable predicate-logic term in tree representation, composed from
    variable names and constant names, and function names applied to them.

    Attributes:
        root (`str`): the constant name, variable name, or function name at the
            root of the term tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a function name.
    """

    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str, arguments: Optional[Sequence[Term]] = None):
        """Initializes a `Term` from its root and root arguments.

        Parameters:
            root: the root for the formula tree.
            arguments: the arguments for the root, if the root is a function
                name.
        """
        if is_constant(root) or is_variable(root):
            assert arguments is None
            self.root = root
        else:
            assert is_function(root)
            assert arguments is not None and len(arguments) > 0
            self.root = root
            self.arguments = tuple(arguments)

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current term.

        Returns:
            The standard string representation of the current term.
        """
        root = self.root
        if is_variable(root) or is_constant(root):
            return root
        assert isinstance(self, (Term, Formula))
        arguments_repr = []
        for arg in self.arguments:
            arguments_repr.append(arg.__repr__())
        return '{}({})'.format(root, ','.join(arguments_repr))

    def __eq__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Term` object that equals the
            current term, ``False`` otherwise.
        """
        return isinstance(other, Term) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Term` object or does not
            equal the current term, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Term, str]:
        """Parses a prefix of the given string into a term.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a term.

        Returns:
            A pair of the parsed term and the unparsed suffix of the string. If
            the given string has as a prefix a constant name (e.g., ``'c12'``)
            or a variable name (e.g., ``'x12'``), then the parsed prefix will be
            that entire name (and not just a part of it, such as ``'x1'``).
        """
        if string:
            head = string[0]

            if is_constant(head) or is_variable(head):
                term_prefix, term_suffix = separate_term(string)
                return Term(term_prefix), term_suffix

            assert is_function(head)
            func_name, func_args, remaining = separate_function_or_relation_name(string)
            inner = func_args[1:-1]
            parsed_args = []

            while inner:
                current_arg, inner = separate_term(inner)
                parsed_args.append(current_arg)
                if inner and inner[0] == ',':
                    inner = inner[1:]

            parsed_terms = []
            for arg in parsed_args:
                parsed_term, _ = Term._parse_prefix(arg)
                parsed_terms.append(parsed_term)

            return Term(func_name, parsed_terms), remaining

    @staticmethod
    def parse(string: str) -> Term:
        """Parses the given valid string representation into a term.

        Parameters:
            string: string to parse.

        Returns:
            A term whose standard string representation is the given string.
        """
        term, _ = Term._parse_prefix(string)

        if not string == str(term):
            print(f'Ensure that the string {string} is a well-formed term')
            assert False

        return term

    def constants(self) -> Set[str]:
        """Finds all constant names in the current term.

        Returns:
            A set of all constant names used in the current term.
        """
        root = self.root

        if is_constant(root):
            return {root}

        if is_variable(root):
            return set()

        if is_function(root):
            collected_constants = set()
            for arg in self.arguments:
                collected_constants.update(Term.constants(arg))
            return collected_constants

    def variables(self) -> Set[str]:
        """Finds all variable names in the current term.

        Returns:
            A set of all variable names used in the current term.
        """
        root = self.root

        if is_variable(root):
            return {root}

        if is_constant(root):
            return set()

        if is_function(root):
            collected_vars = set()
            for arg in self.arguments:
                collected_vars.update(Term.variables(arg))
            return collected_vars

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current term, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current term.
        """
        root = self.root
        collected_functions = set()

        if is_function(root):
            collected_functions.add((root, len(self.arguments)))
            for arg in self.arguments:
                collected_functions.update(Term.functions(arg))

        return collected_functions

    def substitute(
            self,
            substitution_map: Mapping[str, Term],
            forbidden_variables: AbstractSet[str] = frozenset(),
    ) -> Term:
        """Substitutes in the current term, each constant name `construct` or
        variable name `construct` that is a key in `substitution_map` with the
        term `substitution_map`\ ``[``\ `construct`\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The term resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current term are substituted (i.e., those originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from
                `forbidden_variables`.

        Examples:
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'y'})
            f(c,plus(d,x))

            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,y)')}, {'y'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.1


@lru_cache(maxsize=100)  # Cache the return value of is_equality
def is_equality(string: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return string == "="


@lru_cache(maxsize=100)  # Cache the return value of is_relation
def is_relation(string: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return string[0] >= "F" and string[0] <= "T" and string.isalnum()


@lru_cache(maxsize=100)  # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == "~"


@lru_cache(maxsize=100)  # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == "&" or string == "|" or string == "->"


@lru_cache(maxsize=100)  # Cache the return value of is_quantifier
def is_quantifier(string: str) -> bool:
    """Checks if the given string is a quantifier.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a quantifier, ``False`` otherwise.
    """
    return string == "A" or string == "E"


@frozen
class Formula:
    """An immutable predicate-logic formula in tree representation, composed
    from relation names applied to predicate-logic terms, and operators and
    quantifications applied to them.

    Attributes:
        root (`str`): the relation name, equality relation, operator, or
            quantifier at the root of the formula tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a relation name or the
            equality relation.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
        variable (`~typing.Optional`\\[`str`]): the variable name quantified by
            the root, if the root is a quantification.
        statement (`~typing.Optional`\\[`Formula`]): the statement quantified by
            the root, if the root is a quantification.
    """

    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    statement: Optional[Formula]

    def __init__(
            self,
            root: str,
            arguments_or_first_or_variable: Union[Sequence[Term], Formula, str],
            second_or_statement: Optional[Formula] = None,
    ):
        """Initializes a `Formula` from its root and root arguments, root
        operands, or root quantified variable name and statement.

        Parameters:
            root: the root for the formula tree.
            arguments_or_first_or_variable: the arguments for the root, if the
                root is a relation name or the equality relation; the first
                operand for the root, if the root is a unary or binary operator;
                the variable name to be quantified by the root, if the root is a
                quantification.
            second_or_statement: the second operand for the root, if the root is
                a binary operator; the statement to be quantified by the root,
                if the root is a quantification.
        """
        if is_equality(root) or is_relation(root):
            # Populate self.root and self.arguments
            assert isinstance(arguments_or_first_or_variable, Sequence) and not isinstance(
                arguments_or_first_or_variable, str
            )
            if is_equality(root):
                assert len(arguments_or_first_or_variable) == 2
            assert second_or_statement is None
            self.root, self.arguments = root, tuple(arguments_or_first_or_variable)
        elif is_unary(root):
            # Populate self.first
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is None
            self.root, self.first = root, arguments_or_first_or_variable
        elif is_binary(root):
            # Populate self.first and self.second
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is not None
            self.root, self.first, self.second = (
                root,
                arguments_or_first_or_variable,
                second_or_statement,
            )
        else:
            assert is_quantifier(root)
            # Populate self.variable and self.statement
            assert isinstance(arguments_or_first_or_variable, str) and is_variable(arguments_or_first_or_variable)
            assert second_or_statement is not None
            self.root, self.variable, self.statement = (
                root,
                arguments_or_first_or_variable,
                second_or_statement,
            )

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        root = self.root

        if is_equality(root):
            left = self.arguments[0].__repr__()
            right = self.arguments[1].__repr__()
            return '{}{}{}'.format(left, root, right)

        if is_relation(root):
            args = []
            for arg in self.arguments:
                args.append(arg.__repr__())
            return '{}({})'.format(root, ','.join(args))

        if is_unary(root):
            operand = self.first.__repr__()
            return '{}{}'.format(root, operand)

        if is_binary(root):
            left = self.first.__repr__()
            right = self.second.__repr__()
            return '({}{}{})'.format(left, root, right)

        assert is_quantifier(root)
        stmt = self.statement.__repr__()
        return '{}{}[{}]'.format(root, self.variable, stmt)

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Formula, str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a formula.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a term followed by an equality
            followed by a constant name (e.g., ``'f(y)=c12'``) or by a variable
            name (e.g., ``'f(y)=x12'``), then the parsed prefix will include
            that entire name (and not just a part of it, such as ``'f(y)=x1'``).
        """
        sample = string[0]

        if is_constant(sample) or is_variable(sample) or is_function(sample):
            left_term, rest = separate_term(string)
            assert rest and rest[0] == '=', "Expected '=' after term"
            right_term, rest = separate_term(rest[1:])
            term1 = Term.parse(left_term)
            term2 = Term.parse(right_term)
            formula = Formula('=', [term1, term2])

        elif is_relation(sample):
            relation, args_str, rest = separate_function_or_relation_name(string)
            args_inside = args_str[1:-1]
            terms_list = []

            while args_inside:
                arg_part, args_inside = separate_term(args_inside)
                terms_list.append(arg_part)
                if args_inside and args_inside[0] == ',':
                    args_inside = args_inside[1:]

            parsed_terms = []
            for arg in terms_list:
                term_obj, _ = Term._parse_prefix(arg)
                parsed_terms.append(term_obj)

            formula = Formula(relation, parsed_terms)

        elif is_unary(sample):
            subformula, rest = Formula._parse_prefix(string[1:])
            formula = Formula('~', subformula)

        elif is_quantifier(sample):
            quant = sample
            var_name, after_var = separate_term(string[1:])
            statement_str, after_stmt = separate_bracketed_formula(after_var)
            inner_formula, _ = Formula._parse_prefix(statement_str[1:-1])
            formula = Formula(quant, var_name, inner_formula)
            rest = after_stmt

        else:
            assert sample == '(', "Expected '(' at the start of binary formula"
            enclosed_formula, after_formula = separate_bracketed_formula(string)
            left_operand, mid = Formula._parse_prefix(enclosed_formula[1:])
            operator, right_part = separate_operator(mid)
            right_operand, _ = Formula._parse_prefix(right_part)
            formula = Formula(operator, left_operand, right_operand)
            rest = after_formula

        return formula, rest

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        formula, _ = Formula._parse_prefix(string)

        if string != str(formula):
            print(f'Ensure that the string {string} is a well-formed formula')
            assert False

        return formula

    def constants(self) -> Set[str]:
        """Finds all constant names in the current formula.

        Returns:
            A set of all constant names used in the current formula.
        """
        root = self.root

        if is_quantifier(root):
            return Formula.constants(self.statement)

        if is_unary(root):
            return Formula.constants(self.first)

        if is_binary(root):
            left_constants = Formula.constants(self.first)
            right_constants = Formula.constants(self.second)
            return left_constants.union(right_constants)

        if is_equality(root) or is_relation(root):
            collected_constants = set()
            for arg in self.arguments:
                collected_constants.update(Term.constants(arg))
            return collected_constants

    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        root = self.root

        if is_quantifier(root):
            inner_vars = Formula.variables(self.statement)
            return inner_vars.union({self.variable})

        if is_unary(root):
            return Formula.variables(self.first)

        if is_binary(root):
            left_vars = Formula.variables(self.first)
            right_vars = Formula.variables(self.second)
            return left_vars.union(right_vars)

        if is_equality(root) or is_relation(root):
            collected_vars = set()
            for arg in self.arguments:
                collected_vars.update(Term.variables(arg))
            return collected_vars

    def free_variables(self) -> Set[str]:
        """Finds all variable names that are free in the current formula.

        Returns:
            A set of every variable name that is used in the current formula not
            only within a scope of a quantification on that variable name.
        """
        root = self.root

        if is_quantifier(root):
            inner_free = Formula.free_variables(self.statement)
            return inner_free.difference({self.variable})

        if is_unary(root):
            return Formula.free_variables(self.first)

        if is_binary(root):
            left_free = Formula.free_variables(self.first)
            right_free = Formula.free_variables(self.second)
            return left_free.union(right_free)

        if is_equality(root) or is_relation(root):
            collected_free = set()
            for arg in self.arguments:
                collected_free.update(Term.variables(arg))
            return collected_free

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current formula.
        """
        root = self.root

        if is_quantifier(root):
            return Formula.functions(self.statement)

        if is_unary(root):
            return Formula.functions(self.first)

        if is_binary(root):
            left_funcs = Formula.functions(self.first)
            right_funcs = Formula.functions(self.second)
            return left_funcs.union(right_funcs)

        if is_equality(root) or is_relation(root):
            collected_funcs = set()
            for arg in self.arguments:
                collected_funcs.update(Term.functions(arg))
            return collected_funcs

    def relations(self) -> Set[Tuple[str, int]]:
        """Finds all relation names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of relation name and arity (number of arguments) for
            all relation names used in the current formula.
        """
        root = self.root

        if is_relation(root):
            return {(root, len(self.arguments))}

        if is_quantifier(root):
            return Formula.relations(self.statement)

        if is_unary(root):
            return Formula.relations(self.first)

        if is_binary(root):
            left_relations = Formula.relations(self.first)
            right_relations = Formula.relations(self.second)
            return left_relations.union(right_relations)

        # Если ни одно условие не выполнено
        return set()

    def substitute(
            self,
            substitution_map: Mapping[str, Term],
            forbidden_variables: AbstractSet[str] = frozenset(),
    ) -> Formula:
        """Substitutes in the current formula, each constant name `construct` or
        free occurrence of variable name `construct` that is a key in
        `substitution_map` with the term
        `substitution_map`\ ``[``\ `construct`\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The formula resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current formula are substituted (i.e., those originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from `forbidden_variables`
                or a variable name occurrence that becomes bound when that term
                is substituted into the current formula.

        Examples:
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'z'})
            Ay[c=plus(d,x)]

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,z)')}, {'z'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: z

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,y)')})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.2

    def propositional_skeleton(
            self,
    ) -> Tuple[PropositionalFormula, Mapping[str, Formula]]:
        """Computes a propositional skeleton of the current formula.

        Returns:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation name, equality, or quantifier at its
            root with a propositional variable name, consistently such that
            multiple identical such (outermost) subformulas are substituted with
            the same propositional variable name. The propositional variable
            names used for substitution are obtained, from left to right
            (considering their first occurrence), by calling
            `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``.
            The second element of the pair is a mapping from each propositional
            variable name to the subformula for which it was substituted.

        Examples:
            >>> formula = Formula.parse('((Ax[x=7]&x=7)|(~Q(y)->x=7))')
            >>> formula.propositional_skeleton()
            (((z1&z2)|(~z3->z2)), {'z1': Ax[x=7], 'z2': x=7, 'z3': Q(y)})
            >>> formula.propositional_skeleton()
            (((z4&z5)|(~z6->z5)), {'z4': Ax[x=7], 'z5': x=7, 'z6': Q(y)})
        """
        # Task 9.8

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula, substitution_map: Mapping[str, Formula]) -> Formula:
        """Computes a predicate-logic formula from a propositional skeleton and
        a substitution map.

        Arguments:
            skeleton: propositional skeleton for the formula to compute,
                containing no constants or operators beyond ``'~'``, ``'->'``,
                ``'|'``, and ``'&'``.
            substitution_map: mapping from each propositional variable name of
                the given propositional skeleton to a predicate-logic formula.

        Returns:
            A predicate-logic formula obtained from the given propositional
            skeleton by substituting each propositional variable name with the
            formula mapped to it by the given map.

        Examples:
            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z1&z2)|(~z3->z2))'),
            ...     {'z1': Formula.parse('Ax[x=7]'), 'z2': Formula.parse('x=7'),
            ...      'z3': Formula.parse('Q(y)')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))

            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z9&z2)|(~z3->z2))'),
            ...     {'z2': Formula.parse('x=7'), 'z3': Formula.parse('Q(y)'),
            ...      'z9': Formula.parse('Ax[x=7]')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))
        """
        for operator in skeleton.operators():
            assert is_unary(operator) or is_binary(operator)
        for variable in skeleton.variables():
            assert variable in substitution_map
        # Task 9.10
