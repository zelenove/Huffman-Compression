"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes

def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos) for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    bytes_dict = {}
    for i in text:
        if i in bytes_dict:
            bytes_dict[i] += 1
        else:
            bytes_dict[i] = 1
    return bytes_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq2 = {24: 1}
    >>> t2 = huffman_tree(freq2)
    >>> t2 == HuffmanNode(24)
    True
    """
    # Create a list of frequency tuples given the items in freq_dict in the
    # following format: [(int, HuffmanNode())]
    frequency_tuples = []
    for item, frequency in freq_dict.items():
        frequency_tuples.append((frequency, HuffmanNode(item)))

    # List of tuples sorted by a key item, in this case the frequency
    # the key is defined in the function getkey()
    frequency_tuples = sorted(frequency_tuples, key=getkey)

    while len(frequency_tuples) > 1:
        lowest_freq, lowest_node = frequency_tuples[0]
        lowest_freq_2, lowest_node_2 = frequency_tuples[1]
        # Lowest frequency node goes on the right
        freq_tup = (lowest_freq + lowest_freq_2,
                    (HuffmanNode(None, lowest_node, lowest_node_2)))

        # Remove the two nodes use to make the new node
        frequency_tuples.remove(frequency_tuples[0])
        frequency_tuples.remove(frequency_tuples[0])
        # Append the new node and resort frequency_tuples
        frequency_tuples.append(freq_tup)
        frequency_tuples = sorted(frequency_tuples, key=getkey)

    return frequency_tuples[0][1]


def getkey(item):
    """ Define what composition key we want to use for sorted()

    For a fruequency tuple defined in huffman_tree, item[0] is frequency

    @param tuple item: tuple represenation of dictionary value
    @rtype: int

    >>> l = [(3, 'b'), (26, 'd'), (1, 'a'), (15, 'c')]
    >>> sorted(l, key=getkey)
    [(1, 'a'), (3, 'b'), (15, 'c'), (26, 'd')]
    """
    return item[0]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    # Since Nonetype is not an acceptable parameter, we skip this base case
    if tree.left is None and tree.right is None:
        return {tree.symbol: ""}

    else:
        paths = {}

        left = get_codes(tree.left)
        for path in left:
            paths[path] = "0" + left[path]

        right = get_codes(tree.right)
        for path in right:
            paths[path] = "1" + right[path]

        return paths


def number_nodes_helper(tree, current_num):
    """Recursive helper for number_nodes() that returns the next number to be
    used and has a parameter to check what the current number is

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param int current_num: the number of the next node
    @rtype: int

    >>> number_nodes_helper(HuffmanNode(None), 0)
    0
    >>> freq = {"k":2}
    >>> number_nodes_helper(huffman_tree(freq), 0)
    0
    >>> number_nodes_helper(HuffmanNode(None, HuffmanNode(3), HuffmanNode(2)),0)
    1
    """
    if not tree or tree.is_leaf():
        return current_num

    else:
        # If number returned is not changed, tree.left is either a leaf or node
        returned_left = number_nodes_helper(tree.left, current_num)
        if returned_left == current_num:
            returned_right = number_nodes_helper(tree.right, current_num)
            # if tree.right is leaf or node, then tree.num becomes the number
            if returned_right == current_num:
                tree.number = current_num
                return current_num + 1
            # Otherwise tree.number is one greater than the right recursive call
            else:
                tree.number = returned_right
                return tree.number + 1

        returned_right = number_nodes_helper(tree.right, returned_left)

        if returned_right == returned_left:
            tree.number = returned_left
            return tree.number + 1

        # If both recursive call are successful
        tree.number = returned_right
        return tree.number + 1


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # A recursive helper is used since number_nodes does not provide enough
    # parameters and does not return anything
    number_nodes_helper(tree, 0)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes = get_codes(tree)

    total_bits = 0
    total_freq = 0

    # Iterate through code_dict and multiply length by frequency in freq_dict
    # also keep a tally of total frequency
    for c in codes:
        code_len = len(codes[c])
        freq = freq_dict[c]
        total_bits += (code_len * freq)
        total_freq += freq

    return total_bits/total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # All the codes are stored in sequential order as a series of string bits
    all_codes = []
    for symbol in text:
        code = codes[symbol]
        for bit in code:
            all_codes.append(bit)

    octets = []
    if len(all_codes) > 8:  # More than one octet in the list of codes:
        # How many bits will remain after all the octets are created
        offset = len(all_codes) % 8
        # Length of the bits that can be made into octets
        octet_len = len(all_codes) - offset

        # 8 is used for the stop index because range() is from start:stop-1
        start_index = 0
        stop_index = 8

        # Loop will iterate until the first index is equal to or above the
        # index of the last bit used in a full octet
        while start_index < octet_len:
            code = ""
            for index in range(start_index, stop_index):
                code += all_codes[index]
            octets.append(bits_to_byte(code))
            start_index = stop_index
            stop_index += 8

        # Whatever bits remain are made into a string of length 1-7
        less_octet = ""
        for i in range(octet_len, len(all_codes)):
            less_octet += all_codes[i]
        octets.append(bits_to_byte(less_octet))

    else:  # No octets can be maade since len(all_codes) < 8
        less_octet = ""
        for i in range(0, len(all_codes)):
            less_octet += all_codes[i]

        octets.append(bits_to_byte(less_octet))

    return bytes(octets)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    # The recursive helper function return a list of ints which must be turned
    # into bytes
    lst = tree_to_bytes_helper(tree)
    return bytes(lst)


def tree_to_bytes_helper(tree):
    """Postorder traversal of Huffman tree rooted at tree
    returns a list of numbers representing the interior node's configuration
    as described in the handout

    @param HuffmanNode tree: Huffman tree rooted at node 'tree'
    @rtype: list[int]

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> tree_to_bytes_helper(tree)
    [0, 3, 0, 2]
    """
    lst = []
    if not tree or tree.is_leaf():
        return lst
    else:
        lst += tree_to_bytes_helper(tree.left)
        lst += tree_to_bytes_helper(tree.right)

        # If subtree is a leaf return append 0 and symbol of the leaf
        if tree.left.is_leaf():
            lst.append(0)
            lst.append(tree.left.symbol)

        else:  # Otherwise, append 1 and the number of the subtree
            lst.append(1)
            lst.append(tree.left.number)

        # Repeat statements for right subtree
        if tree.right.is_leaf():
            lst.append(0)
            lst.append(tree.right.symbol)

        else:
            lst.append(1)
            lst.append(tree.right.number)

        return lst


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root_node = node_lst[root_index]
    left_not_leaf = root_node.l_type
    right_not_leaf = root_node.r_type
    l_symbol = root_node.l_data
    r_symbol = root_node.r_data

    # Base case, tree is an interior node with two leaves
    if not left_not_leaf and not right_not_leaf:
        return HuffmanNode(None, HuffmanNode(l_symbol), HuffmanNode(r_symbol))

    # Left tree is an interior node, right tree is a leaf
    elif left_not_leaf and not right_not_leaf:
        left_index = root_node.l_data
        left = generate_tree_general(node_lst, left_index)

        return HuffmanNode(None, left, HuffmanNode(r_symbol))

    # Right tree is an interior node, left tree is a leaf
    elif right_not_leaf and not left_not_leaf:
        right_index = root_node.r_data
        right = generate_tree_general(node_lst, right_index)

        return HuffmanNode(None, HuffmanNode(l_symbol), right)

    # Both subtrees are interior nodes
    else:
        left_index = root_node.l_data
        right_index = root_node.r_data

        left = generate_tree_general(node_lst, left_index)
        right = generate_tree_general(node_lst, right_index)

        return HuffmanNode(None, left, right)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    root_node = node_lst[root_index]
    r = root_index - 1
    left_is_node = root_node.l_type
    right_is_node = root_node.r_type

    # Base case: both subtrees are leaves
    if not left_is_node and not right_is_node:
        left_dat = root_node.l_data
        right_dat = root_node.r_data
        return HuffmanNode(None, HuffmanNode(left_dat), HuffmanNode(right_dat))

    # If right tree is a leaf, left tree starts at root_index - 1 (r for short)
    elif left_is_node and not right_is_node:
        right_dat = root_node.r_data
        left = generate_tree_postorder(node_lst, r)
        return HuffmanNode(None, left, HuffmanNode(right_dat))

    # If the right tree is a node and left tree is a leaf
    elif not left_is_node and right_is_node:
        right = generate_tree_postorder(node_lst, r)
        nodes_created = internal_node_quantity(right)
        r -= nodes_created
        left_dat = root_node.l_data
        return HuffmanNode(None, HuffmanNode(left_dat), right)

    # If right tree is not a leaf, left tree starts at (root_index - 1 - n )
    # where n is the number of nodes created by the right recursive call
    else:
        right = generate_tree_postorder(node_lst, r)
        nodes_created = internal_node_quantity(right)

        r -= nodes_created
        left = generate_tree_postorder(node_lst, r)

        return HuffmanNode(None, left, right)


def internal_node_quantity(tree):
    """ Return how many internal nodes a huffman tree has in order to increment
    a list in preorder traversal

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @rtype: int

    >>> tree = HuffmanNode(None, HuffmanNode(1), HuffmanNode(2))
    >>> internal_node_quantity(tree)
    1
    >>> tree2 = HuffmanNode(None, HuffmanNode(5), tree)
    >>> internal_node_quantity(tree2)
    2
    """
    if tree.is_leaf():
        return 0

    else:
        count = 1
        count += internal_node_quantity(tree.left)
        count += internal_node_quantity(tree.right)
        return count


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    Doctest from piazza:
    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))), HuffmanNode(None, \
HuffmanNode(4), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> result = generate_uncompressed(t, text, size)
    >>> result == bytes([5, 3, 1, 1])
    True

    >>> d = make_freq_dict(bytes([1, 2, 1, 1, 1, 1, 2, 2, 4, 6]))
    >>> tree = huffman_tree(d)
    >>> code = get_codes(tree)
    >>> result = generate_compressed(bytes([1, 2, 1, 1, 1, 1, 2, 2, 4, 6]),\
     code)
    >>> a = generate_uncompressed(tree, result, 11)
    """
    # Freq_dict with key and value reversed for easier access
    codes = get_codes(tree)
    reverse_codes = {}
    for x, y in codes.items():
        reverse_codes[y] = x

    # All of the compressed bits in a list
    bit_stream = []
    for i in range(0, len(text)):
        bit_stream += byte_to_bits(text[i])

    symbols = []
    # Controls if a longer code is being constructed
    j = None
    index = 0
    while len(symbols) < size:
        if j is None:
            i = bit_stream[index]
            index += 1

            if i in reverse_codes:
                symbols.append(reverse_codes[i])
            else:
                # A longer code will be built
                j = i + bit_stream[index]
                index += 1
        else:  # Check if code is in dictionary or needs to be longer
            if j in reverse_codes:
                symbols.append(reverse_codes[j])
                j = None

            else:
                j += bit_stream[index]
                index += 1

    return bytes(symbols)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # Frequency tuples are created for easier access and sorting
    frequency_tuples = []
    for item, frequency in freq_dict.items():
        frequency_tuples.append((frequency, item))
    frequency_tuples = sorted(frequency_tuples, key=getkey)

    # List will be used as an FIFO container for inorder traversal
    queue = [tree]

    while queue != []:
        curr_node = queue.pop(0)
        # curr_node corresponds to the most frequent leaf position so far
        if curr_node.is_leaf():
            most_freq = frequency_tuples[-1][1]
            if curr_node.symbol != most_freq:
                curr_node.symbol = most_freq
            del frequency_tuples[-1]

        # Subtrees are added to the queue
        else:
            if curr_node.right:
                queue.append(curr_node.right)
            if curr_node.left:
                queue.append(curr_node.left)


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
