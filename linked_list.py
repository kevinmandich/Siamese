## linked_list.py

class FIFO(object):

    def __init__(self):
        
        self.nextIn = 0
        self.nextOut = 0
        self.data = {}

    def append(self, value):

        self.data[self.nextIn] = value
        self.nextIn += 1

        return None

    def remove(self):

        try:
            self.data.pop(self.nextOut)
            self.nextOut += 1
        except:
            print 'Can\'t remove the last element because there are no '\
                 +'elements left in the data structure'

        return None


class Node1(object):

    def __init__(self, value):

        self.value = value
        self.nextNode = None


class LinkedList(object):

    def __init__(self, head=None):

        self.head = head

    def __str__(self):

        currentNode = self.head
        ll = str(currentNode.value) + ' '
        while currentNode.nextNode:
            ll += str(currentNode.nextNode.value) + ' '
            currentNode = currentNode.nextNode

        return ll.strip()

    def insert(self, node):

        if not self.head:
            self.head = node
        else:
            node.nextNode = self.head
            self.head = node

    def search_recursive(self, linkedList, node):

        if node == linkedList.head:
            return linkedList.head
        else:
            if linkedList.head.nextNode:
                return self.search_recursive(LinkedList(linkedList.head.nextNode), node)
            else:
                raise ValueError('This node doesn\'t exist in the Linked List')

    def search(self, node):

        currentNode = self.head
        while currentNode.nextNode:
            if currentNode == node:
                return currentNode
            currentNode = currentNode.nextNode
        raise ValueError('This node doesn\'t exist in the Linked List')

    def size(self):

        currentNode = self.head
        size = 0
        while currentNode.nextNode:
            size += 1
            currentNode = currentNode.nextNode

        return size

    def delete(self, node):

        if self.size() == 0:
            raise ValueError('Linked list is empty')

        currentNode = self.head
        previousNode = None
        found = None
        while not found:
            if node == currentNode:
                found = True
            elif currentNode is None:
                raise ValueError('This node doesn\'t exist in the Linked List')
            else:
                previousNode = currentNode
                currentNode = currentNode.nextNode
        if previousNode is None:
            self.head = currentNode.nextNode
        else:
            previousNode.nextNode = currentNode.nextNode











class Node2(object):

    def __init__(self, value):

        self.value = value
        self.nextNode = None
        self.prevNode = None


class DoublyLinkedList(object):

    def __init__(self, head=None):

        self.head = head
        self.tail = None

    def __str__(self):

        if self.size() == 0:
            return 'The list is empty'
        else:
            currentNode = self.head
            dll = str(currentNode.value) + ' '
            while currentNode.nextNode:
                dll += str(currentNode.nextNode.value) + ' '
                currentNode = currentNode.nextNode
            return dll.strip()

    def size(self):

        if not self.head:
            return 0

        size = 0
        currentNode = self.head
        while currentNode.nextNode:
            size += 1
            currentNode = currentNode.nextNode

        return size

    def search(self, node):

        if not self.head:
            raise ValueError('Can\'t search an empty list')

        currentNode = self.head
        while currentNode.nextNode:
            if node == currentNode:
                return currentNode
            currentNode = currentNode.nextNode

        raise ValueError('Node does not exist in the linked list.')

    def insert_beginning(self, node):

        if not self.head:
            self.head = node
        else:
            if not self.tail:
                node.nextNode = self.head
                self.tail = self.head
                self.tail.prevNode = node
                self.head = node
            else:
                node.nextNode = self.head
                self.head.prevNode = node
                self.head = node

    def insert_end(self, node):

        if not self.head:
            self.head = node
        else:
            if not self.tail:
                node.prevNode = self.head
                self.head.nextNode = node
                self.tail = node
            else:
                self.tail.nextNode = node
                node.prevNode = self.tail
                self.tail = node






if __name__ == '__main__':

    fifo = FIFO()

    ll = LinkedList()
    ll.insert(Node1(5))
    for x in range(10,16):
        ll.insert(Node1(x))

    sn = ll.search_recursive(ll, ll.head.nextNode.nextNode.nextNode)
    sn2 = ll.search(ll.head.nextNode.nextNode.nextNode)

    d1 = DoublyLinkedList()
    d1.insert_beginning(Node2(5))
    d1.insert_beginning(Node2(8))
    d1.insert_beginning(Node2(18))

    d2 = DoublyLinkedList()
    d2.insert_end(Node2(6))
    d2.insert_end(Node2(7))
    d2.insert_end(Node2(8))

    d3 = DoublyLinkedList()
    d3.insert_end(Node2(10))
    d3.insert_beginning(Node2(20))

    d4 = DoublyLinkedList()
    d4.insert_beginning(Node2(50))
    d4.insert_end(Node2(100))


















        
