import numpy as np


class HeapSort(object):
    __count = 0
    __capacity = 30
    arr = [0] * __capacity

    def __init__(self):
        pass

    def get_capacity(self):
        return self.__capacity

    def extend_capacity(self):
        self.__capacity += 30
        self.arr.extend([0] * 30)

    def get_count(self):
        return self.__count

    def init_heap(self, nums):
        while self.get_capacity() + 1 < len(nums):
            self.extend_capacity()
            print("扩容+30 成功")
        print(self.arr)
        for i, item in enumerate(nums):
            self.arr[i + 1] = item
            self.__count += 1
        print("count is {}".format(self.__count))
        print('original is :')
        print(self.arr)
        self.adjust()
        print('initialized results:')
        print(self.arr)

    def shift_down(self,pos):
        j = 2 * pos
        while j <= self.get_count():
            if j + 1 <= self.get_count():
                if self.arr[j+1] > self.arr[j]:
                    j = j + 1
            if self.arr[pos] < self.arr[j]:
                self.arr[pos] , self.arr[j] = self.arr[j] , self.arr[pos]
                pos = j
                j = 2 * j
            else:
                break

    def adjust(self):
        n = self.get_count() // 2
        while n >= 1:
            self.shift_down(n)
            n -= 1

    def is_big_heap(self):
        n = self.get_count() // 2
        while n >= 1:
            if self.arr[n] >= self.arr[2 * n]:
                if 2 * n + 1 <= self.get_count():
                    if self.arr[n] >= self.arr[2*n+1]:
                        n -= 1
                        continue
                    else: return False
                n -= 1
            else: return False
        return True

    def get_max(self):
        max_num = self.arr[1]

        self.arr[1], self.arr[self.get_count()] = self.arr[self.get_count()], self.arr[1]
        self.__count -= 1
        self.adjust()
        return max_num

    def heap_sort(self):
        res = [0] * self.get_count()
        while self.get_count() > 0:
            res.append(self.get_max())
            print('heap sort result is: ')
            print(res[::-1])

    def insert(self,num):
        if self.__capacity <= self.__count:
            self.extend_capacity()
            self.__count = self.get_count() + 1
            self.arr[self.__count] = num
            self.arr[self.__count] , self.arr[1] = self.arr[1] , self.arr[self.__count]
            self.adjust()
            print("after insert: \n{}".format(self.arr))

    def print_heap(self):
        print("heap status is: ")
        print(self.arr)


if __name__ == '__main__':
    nums = np.random.randint(0,100,10)
    print(nums)
    heap1 = HeapSort()
    heap1.init_heap(nums)
    heap1.insert(55)
    print('is big heap: {}'.format(heap1.is_big_heap()))
    # heap1.heap_sort()
    heap1.print_heap()










