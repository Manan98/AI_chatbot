final = []

for i in range(0, 7972, 2):
    l1 = "L" + str(i)
    l2 = "L" + str(i+1)
    var = "u0 +++$+++ u1 +++$+++ m0 +++$+++ ['" + l1 + "', '" + l2 + "']"
    final.append(var)

table2 = open('table2.txt', 'w')
for item in final:
    table2.write("%s\n" % item)