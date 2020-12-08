#######################################
# CONSTANTS
#######################################
import string


#######################################
# CONSTANTS
#######################################
DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS
CHARACTERS = string.printable


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
        result = f'{self.error_name}: {self.details}'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln+1}'
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


#######################################
# POSTION
#######################################
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char):
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
TT_INT          = 'INTEGER'
TT_FLOAT        = 'FLOAT'
TT_BOOL         = 'BOOLEAN'
TT_CHAR         = 'CHARACTER'
TT_STR          = 'STRING'
TT_PLUS         = 'PLUS'
TT_MINUS        = 'MINUS'
TT_MUL          = 'MULTIPLY'
TT_DIV          = 'DIVIDE'
TT_MOD          = 'MODULO'
TT_LPAREN       = 'LPAREN'
TT_RPAREN       = 'RPAREN'



class Token:
    def __init__(self, _type, value=None):
        self._type = _type
        self.value = value

    def __repr__(self):
        if self.value:
            return f'{self._type}:{self.value}'
        return f'{self._type}'



#######################################
# LEXER
#######################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0,-1, fn, text)
        self.current_char = None
        self.advance()

    #######################################
    # ADVANCE
    #######################################
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None



    #######################################
    # MAKE TOKENS
    #######################################
    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()

            elif self.current_char in DIGITS:
                tokens.append(self.make_number())


            elif self.current_char == "'":
                tokens.append(self.make_char())

            elif self.current_char == '"':
                tokens.append(self.make_string())
                self.advance()

            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.advance()

            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.advance()

            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.advance()

            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.advance()

            elif self.current_char == '%':
                tokens.append(Token(TT_MOD))
                self.advance()

            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.advance()

            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.advance()

            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start,self.pos,"'"+char+"'")

        return tokens, None


    #######################################
    # MAKE NUMBER
    #######################################
    def make_number(self):
        num_str = ''
        dot_count = 0

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break

                dot_count += 1
                num_str += '.'

            else:
                num_str += self.current_char

            self.advance()


        if dot_count == 0:
            return Token(TT_INT, int(num_str))

        else:
            return Token(TT_FLOAT, float(num_str))


    #######################################
    # MAKE CHARACTER
    #######################################
    def make_char(self):
        char_str = ''
        str_count = 0

        while self.current_char != None and self.current_char in CHARACTERS:
            if self.current_char == "'":
                char_str += self.current_char
                str_count += 1

            else:
                if str_count != 0 and str_count <3:
                    char_str += self.current_char
                    str_count+= 1

                else:
                    break
            self.advance()

        if str_count == 3:
            return Token(TT_CHAR, char_str)



    #######################################
    # MAKE STRING
    #######################################
    def make_string(self):
        str_str = ''
        quote_count = 0

        while self.current_char != None and self.current_char in CHARACTERS:
            if self.current_char == '"':
                str_str += self.current_char
                quote_count += 1

                if quote_count == 2:
                    break

            else:
                str_str += self.current_char

            self.advance()


        return Token(TT_STR, str_str)



#######################################
# RUN
#######################################
def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()

    return tokens, error



