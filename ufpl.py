#######################################
# IMPORTS
#######################################

from strings_with_arrows import *

import string

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


#######################################
# ERRORS
#######################################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)


class InvalidInstantiationError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Instantiation', details)


class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result


#######################################
# POSITION
#######################################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


#######################################
# TOKENS
#######################################

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_CHAR = 'CHAR'
TT_BOOL = 'BOOL'
TT_STRING = 'STRING'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_CONCAT = 'CONCAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_EQ = 'EQ'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EE = 'EE'  #
TT_NE = 'NE'  #
TT_LT = 'LT'  #
TT_GT = 'GT'  #
TT_LTE = 'LTE'  #
TT_GTE = 'GTE'  #
TT_EOF = 'EOF'
TT_EOS = 'EOS'

KEYWORDS = [
    'VAR',
    'AND',
    'OR',
    'NOT',
    'START',
    'END',
    'AS',
    'INT',
    'BOOL',
    'CHAR'
]


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


#######################################
# LEXER
#######################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        statements = []
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char == '\n':
                tokens.append(Token(TT_EOS, pos_start=self.pos))
                statements.append(tokens)
                tokens = []
                while self.current_char == '\n':
                    self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == ',':
                self.advance()
                while self.current_char == ' ':
                    self.advance()
                currpos = self.pos.copy()
                chck = self.multi_identifier()
                if isinstance(chck, Token):
                    tokens.append(chck)
                else:
                    self.advance()
                    return [], InvalidSyntaxError(currpos, self.pos, "Invalid Identifier Name")

            elif self.current_char == "'":
                tokens.append(self.make_character())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '&':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                if self.pos.col == 0:
                    self.skip_comment()
                else:
                    tokens.append(Token(TT_MUL, pos_start=self.pos))
                    self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                token, error = self.make_not_equals()
                if error: return [], error
                tokens.append(token)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        statements.append(tokens)
        return statements, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
            num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        self.advance()

        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '[':
                self.advance()
                string += self.current_char
                self.advance()
                if self.current_char == ']':
                    self.advance()
                    continue
                else:
                    break
            elif self.current_char == '#':
                string += '\n'
                self.advance()
            string += self.current_char
            self.advance()

        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)

    def make_character(self):
        char = ''
        pos_start = self.pos.copy()
        escape_character = False
        self.advance()
        isChar = False

        while self.current_char is not None and self.current_char != "'":
            if isChar == False:
                char += self.current_char
                isChar = True
                self.advance()
            else:
                break
        self.advance()
        return Token(TT_CHAR, char, pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def multi_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        if id_str in KEYWORDS:
            return 0
        return Token(TT_IDENTIFIER, id_str, pos_start, self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

    def make_equals(self):
        tok_type = TT_EQ
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_EE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = TT_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_LTE
        elif self.current_char == '>':
            self.advance()
            tok_type = TT_NE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_GTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def skip_comment(self):
        self.advance()
        while self.current_char != '\n':
            self.advance()
        self.advance()


#######################################
# NODES
#######################################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class StringNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class addStringNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class CharNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class BoolNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class VarAssignNode:
    def __init__(self, var_name_tok, value_node, datatype):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.data_type = datatype

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.data_type.pos_end


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'


#######################################
# PARSE RESULT
#######################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self


#######################################
# PARSER
#######################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self, ):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = ParseResult()
        if self.current_tok.matches(TT_KEYWORD, 'START') or self.current_tok.matches(TT_KEYWORD, 'END'):
            current = self.current_tok.value
            self.advance()
            if self.current_tok.type != TT_EOS and self.current_tok.type != TT_EOF:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "does not support 2 or more statements in one line"
                ))
            return current

        batch = self.expr()
        if isinstance(batch, list):
            for res in batch:
                if not res.error and (self.current_tok.type != TT_EOF and self.current_tok.type != TT_EOS):
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
                    ))

        return batch

    ###################################

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok))

        elif tok.type == TT_CHAR:
            res.register_advancement()
            self.advance()
            return res.success(CharNode(tok))

        elif tok.type == TT_BOOL:
            res.register_advancement()
            self.advance()
            return res.success(BoolNode(tok))

        elif tok.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Expected int, float, identifier, '+', '-', '('"
        ))

    def power(self):
        return self.bin_op(self.atom, (TT_POW,), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, 'NOT'):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected int, float, identifier, '+', '-', '(' or 'NOT'"
            ))

        return res.success(node)

    def expr(self):
        res = ParseResult()
        if self.current_tok.type == TT_IDENTIFIER and self.current_tok.pos_start.col == 0 and isstart == False:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Invalid variable position"
            ))


        if self.tok_idx == 0 and self.current_tok.matches(TT_KEYWORD, 'VAR') and isstart == True:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "You can only declare variable before the keyword 'START'"
            ))

        if self.current_tok.matches(TT_KEYWORD, 'VAR'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))

            var_name = []
            batch = list()
            while self.current_tok.type == TT_IDENTIFIER:
                var_name.append(self.current_tok)
                res.register_advancement()
                self.advance()

            if self.current_tok.type != TT_EQ:
                if self.current_tok.type == TT_KEYWORD:
                    if self.current_tok.value == 'AS':
                        res.register_advancement()
                        self.advance()
                        if self.current_tok.type == TT_KEYWORD:
                            data_types = ["INT", "CHAR", "BOOL"]
                            if self.current_tok.value in data_types:
                                datatype = self.current_tok
                                res.register_advancement()
                                self.advance()
                                for i in var_name:
                                    g = ParseResult().success(VarAssignNode(i, None, datatype))
                                    batch.append(g)
                                return batch
                        else:
                            return res.failure(InvalidSyntaxError(
                                self.current_tok.pos_start, self.current_tok.pos_end,
                                "Expected KEYWORD 'INT', 'BOOL', OR 'CHAR'"
                            ))
                    else:
                        return res.failure(InvalidSyntaxError(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected KEYWORD 'AS'"
                        ))
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected KEYWORD"
                    ))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res

            if self.current_tok.type == TT_KEYWORD and self.current_tok.value == 'AS':
                res.register_advancement()
                self.advance()
                data_types = ["INT", "CHAR", "BOOL"]
                if self.current_tok.value in data_types:
                    datatype = self.current_tok
                    res.register_advancement()
                    self.advance()
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected KEYWORD 'INT', 'BOOL', OR 'CHAR'"))
            else:
                if self.current_tok.type == 'CHAR':
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start,
                                                          self.current_tok.pos_end,
                                                          "'' only takes 1 symbol"))
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start,
                                                      self.current_tok.pos_end,
                                                      "Expected KEYWORD 'AS'"))
            for i in var_name:
                batch.append(ParseResult().success(VarAssignNode(i, expr, datatype)))
            return batch

        node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'AND'), (TT_KEYWORD, 'OR'))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'VAR', int, float, identifier, '+', '-', '(' or 'NOT'"
            ))

        return res.success(node)

    ###################################

    def bin_op(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)


#######################################
# RUNTIME RESULT
#######################################

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


#######################################
# VALUES
#######################################

class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        return None, self.illegal_operation(other)

    def subbed_by(self, other):
        return None, self.illegal_operation(other)

    def multed_by(self, other):
        return None, self.illegal_operation(other)

    def dived_by(self, other):
        return None, self.illegal_operation(other)

    def powed_by(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self, other):
        return None, self.illegal_operation(other)

    def execute(self, args):
        return RTResult().failure(self.illegal_operation())

    def copy(self):
        raise Exception('No copy method defined')

    def is_true(self):
        return False

    def illegal_operation(self, other=None):
        if not other: other = self
        return RTError(
            self.pos_start, other.pos_end,
            'Illegal operation',
            self.context
        )


class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number) and isinstance(self, Number):
            if isinstance(other.value, int) and isinstance(self.value, int):
                return Number(self.value + other.value).set_context(self.context), None
            else:
                return Number(str(self.value) + str(other.value)).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Division by zero',
                    self.context
                )

            return Number(self.value / other.value).set_context(self.context), None

    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            if self.value == other.value:
                return Bool("TRUE").set_context(self.context), None
            return Bool("FALSE").set_context(self.context), None
            # Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)


class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def concat_to(self, other):
        return String(self.value + str(other.value)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'"{self.value}"'


class Char(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        return String(self.value + str(other.value)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'"{self.value}"'


class Bool(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        return String(self.value + str(other.value)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'"{self.value}"'


#######################################
# CONTEXT
#######################################

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value, datatype):
        self.symbols[name] = value, datatype

    def remove(self, name):
        del self.symbols[name]


#######################################
# INTERPRETER
#######################################

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    ###################################

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(context).set_pos(node.pos_start,
                                                                node.pos_end)
        )

    def visit_CharNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(context).set_pos(node.pos_start,
                                                                node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)[0]

        if not value:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        data_type = node.data_type.value
        node_dtype = node.value_node
        print(node_dtype)

        if node.value_node == None:
            if data_type == 'INT':
                context.symbol_table.set(var_name, Number(0), data_type)
                return res.success(Number(0))
            elif data_type == 'BOOL':
                context.symbol_table.set(var_name, Bool("False"), data_type)
                return res.success(Bool("False"))
            elif data_type == 'CHAR':
                context.symbol_table.set(var_name, Char(''), data_type)
                return res.success(Char(''))

        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        if data_type == 'INT':
            if not isinstance(value.value, int):
                res.failure(InvalidInstantiationError(
                    value.pos_start, value.pos_end,
                    "Invalid datatype: needs <int>"
                ))

        elif data_type == 'BOOL':
            if value.value not in ("TRUE", "FALSE"):
                res.failure(InvalidInstantiationError(
                    value.pos_start, value.pos_end,
                    "Invalid datatype: needs <TRUE or FALSE>"
                ))

        elif data_type == 'CHAR':
            print(value)
            if not isinstance(value.value, str) or 1 < len(value.value):
                res.failure(InvalidInstantiationError(
                    value.pos_start, value.pos_end,
                    "Invalid datatype: needs <CHAR>"
                ))

        context.symbol_table.set(var_name, value, data_type)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res

        if node.op_tok.type == TT_CONCAT:
            result, error = left.concat_to(right)
        elif node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW:
            result, error = left.powed_by(right)
        elif node.op_tok.type == TT_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == TT_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == TT_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(TT_KEYWORD, 'AND'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEYWORD, 'OR'):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))
        elif node.op_tok.matches(TT_KEYWORD, 'NOT'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number(0), 'INT')
global_symbol_table.set("FALSE", Number(0), 'INT')
global_symbol_table.set("TRUE", Number(1), 'INT')


def run(fn, text):
    global isstart
    isstart = False
    nodes = ["hello"]
    # Generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error
    # return tokens, error
    #
    # Generate AST
    nodes = []
    for i in tokens:
        if not i: continue
        parser = Parser(i)
        ast = parser.parse()
        if not isinstance(ast, list):
            if ast == 'START':
                isstart = True
                continue
            elif ast == 'END':
                break
            if ast.error: return None, ast.error

            nodes.append(ast.node)
        else:
            for n in ast:
                if n.error: return None, ast.error
                nodes.append(n.node)

    # return nodes, None

    # Run program
    interpreter = Interpreter()
    context = Context('<code.ufpl>')
    context.symbol_table = global_symbol_table

    results = []
    for i in nodes:
        result = interpreter.visit(i, context)
        if result.error: return None, result.error
        results.append(result)
    return results, None