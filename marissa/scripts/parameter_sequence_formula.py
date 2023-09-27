from marissa.toolbox.tools import tool_general

test = "MOLLI  3(3)3(3)5 b"

test = tool_general.string_stripper(test, [])

out = ""


sequences = {}
sequences["SASHA GRE"] = ["SASHAGRE"]
sequences["SASHA"] = ["SASHA"]
sequences["ShMOLLI"] = ["SHMOLLI"]
sequences["MOLLI 3(3)5 s"] = ["MOLLI335S"]
sequences["MOLLI 3(3)5 b"] = ["MOLLI335"]
sequences["MOLLI 3(3)3(3)5 s"] = ["MOLLI33335S"]
sequences["MOLLI 3(3)3(3)5 b"] = ["MOLLI33335"]
sequences["MOLLI 4(1)3(1)2 s"] = ["MOLLI41312S"]
sequences["MOLLI 4(1)3(1)2 b"] = ["MOLLI41312"]
sequences["MOLLI 5(3)3 s"] = ["MOLLI533S", "MOLLI5S3S3S"]
sequences["MOLLI 5(3)3 b"] = []

str_seq = "["
keys = list(sequences.keys())
for i in range(len(keys) - 1):
    str_seq = str_seq + "\"" + keys[i] + "\" if (" + " or ".join(["\"" + sequences[keys[i]][j] + "\" in sd" for j in range(len(sequences[keys[i]]))]) + ") else "
str_seq = str_seq + "\"" + keys[-1] + "\" for sd in [tool_general.string_stripper(str(dcm[0x0008, 0x103e].value).upper().replace('SAX',''), [])]][0]"

print(str_seq)

a = 0