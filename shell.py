import ufpl

f = open("code.ufpl", "r")
script = f.read()

tokens, error = ufpl.run("<code>", script)
if error:
    print(error.as_string())
else:
    for i in tokens:
        print(i.value)
