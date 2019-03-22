import re

str = r'<div class="nam">中国</div>';
res = re.findall(r'<div class=".*">(.*?)</div>', str);
print(res);

