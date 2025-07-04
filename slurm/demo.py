input_str = """2.5G	./pp0622
1.8G	./pp0452
32K	./pp0605
20K	./pp0636
20K	./pp0366
20K	./pp0365
20K	./pp0323
20K	./pp0312
20K	./pp0269
16K	./pp0375
12K	./pp0658
12K	./pp0653
12K	./pp0649
12K	./pp0644
12K	./pp0639
12K	./pp0635
12K	./pp0634
12K	./pp0594
12K	./pp0593
12K	./pp0570
12K	./pp0559
12K	./pp0504
12K	./pp0472
12K	./pp0433
12K	./pp0405
12K	./pp0386
12K	./pp0370
12K	./pp0362
12K	./pp0361
12K	./pp0338
12K	./pp0337
12K	./pp0311
12K	./pp0305
12K	./pp0275
4.0K	./pp0496
4.0K	./pp0364
"""

# # 方法一：用 split 和 lstrip
# lines = input_str.splitlines()
# result = []
# for line in lines:
#     # 先按制表符分割，取第二部分，再去掉前导的 "./"
#     name = line.split('\t')[1].lstrip("./")
#     result.append(name)

# output = "\n".join(result)
# print(output)

# 方法二：用正则
import re

# 匹配 "./" 后面的所有非空白字符
names = re.findall(r"\./(\S+)", input_str)
print("\n".join(names))