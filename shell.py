import ufpl

program = []
while text := input(">>"):
    if text == "STOP":
        break;
    program.append(text)


for i in program:
    result, error = ufpl.run('<stdin>',i)

    if error: print(error.as_string())
    else: print(result)