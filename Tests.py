def getkey(item):
    """ Thhis function defines what composition key we want to use for sorted()
    """
    return item


if __name__ == '__main__':
    from huffman import make_freq_dict, huffman_tree, get_codes, count_leaves
    from nodes import HuffmanNode

    """  Sorted() by key test
    l = [[2, 3], [6, 7], [3, 34], [24, 64], [1, 43]]
    l = sorted(l, key=getkey)
    print(l)
    """

    """ Test using multiple arguements in for loop
    d = make_freq_dict(bytes([65, 66, 67, 66]))
    for k, v in d.items():
        print(k, 'corresponds to', v)
    """

    # Testing get_codes()
    freq = {2: 6, 3: 4, 4: 13, 5: 17}
    t = huffman_tree(freq)
    d = get_codes(t)
    print(t)
    print(d)

    freqs = {'a': 2}
    d = HuffmanNode()
    print(d)

    print(get_codes(d))

    print(bytes([65, 66, 67, 66]))

    left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    tree = HuffmanNode(None, left, right)
    number_nodes(tree)
