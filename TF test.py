import collections

string = input()
string = string[1:-1]
alist = string.split(',')
print(alist)


############# 标准形式
# alist = [1,2,3,1,null,2,null,null,null,null,null,1,null,null,null]
############

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def creatTree(alist):
    li = []
    for a in alist:  # 创建结点
        if a == 'null':
            node = Node(a)
        else:
            node = Node(int(a))
        li.append(node)
    parentNum = len(li) // 2 - 1
    for i in range(parentNum+1):
        leftIndex = 2 * i + 1
        rightIndex = 2 * i + 2
        if not li[leftIndex].val =='null':
            li[i].left = li[leftIndex]
        if rightIndex < len(li) and not li[rightIndex].val =='null':  # 判断是否有右结点， 防止数组越界
            li[i].right = li[rightIndex]
    return li[0]


# 层次遍历所有的结点
def BFS(root):
    queue, result = [root], []
    while queue:
        node = queue.pop(0)
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result



def findDuplicateSubtrees(root):
    def serialize(root):
        if not root: return '#'
        return str(root.val) + ',' + serialize(root.left) + ',' + serialize(root.right)

    def traverse(root, counter):
        if not root: return []
        res = []
        chain = serialize(root)
        # print(chain)
        counter[chain] += 1
        if counter[chain] == 2: res.append(root)
        res += traverse(root.left, counter) + traverse(root.right, counter)
        return res
    return traverse(root, collections.Counter())

root = creatTree(alist)

ans_out = findDuplicateSubtrees(root)

tmp_res = []
res = []
if not ans_out:
    print('-1')
else:
    size = -1
    for i_ans in ans_out:
        tmp_res.clear()
        tmp_res = BFS(i_ans)
        if len(tmp_res) > size:
            size = len(tmp_res)
            res.clear()
            res = tmp_res.copy()

    res_final ='['
    for r in res:
        res_final+=str(r)+','
    res_final+= 'null]'
    print(res_final)

# for l in ans_out :
#     if  len(l) > size:
#         size = len(l)
#         res.clear()
#         res = l.copy()
# if size == -1 or size == 0:
#     print('-1')
# else:
#     print(res)






