def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    codes = get_codes(tree)
    reverse_codes = {}
    for x, y in codes.items():
        reverse_codes[y] = x

    bit_stream = []
    for i in text:
        bit_stream += byte_to_bits(text[i])

    symbols = []
    j = None
    index = 0
    while len(symbols) < size:
        if j is None:
            i = bit_stream[index]
            index += 1

            if i in reverse_codes:
                symbols.append(reverse_codes[i])
            else:
                j = i + bit_stream[index]
                index += 1
        else:
            if j in reverse_codes:
                symbols.append(reverse_codes[j])
                j = None

            else:
                j += bit_stream[index]
                index += 1

    return bytes(symbols)


def helper(tree, text):
    """
    @type tree: text
    @type text: bytes
    @rtype: code
    """



if __name__ == "__main__":
    from huffman import *
    from ex8 import *

    # with open("book2.txt.huf", "rb") as f:
    #     num_nodes = f.read(1)[0]
    #     buf = f.read(num_nodes * 4)
    #     node_lst = bytes_to_nodes(buf)
    #     tree = generate_tree_general(node_lst, num_nodes - 1)
    #
    # with open("book2.txt.huf", "rb") as f1:
    #     text = f1.read()
    # freq = make_freq_dict(text)

    # with open("book2.txt", "rb") as f1:
    #     text = f1.read()
    # freq = make_freq_dict(text)
    # tree = huffman_tree(freq)
    # codes = get_codes(tree)
    # comp = generate_compressed(text, codes)
    #
    # result = generate_uncompressed(tree, comp, len(text))
    #
    # print(result)

    # d = make_freq_dict(bytes([1, 2, 1, 0]))
    # codes = get_codes(tree)
    # for du, dy in codes.items():
    #     print(str("{} correspponds to {}").format(du, dy))
    #
    # text = bytes([1, 2, 1, 0])
    # print(text)

    # d = {0: "0", 1: "10", 2: "11"}
    # text = bytes([1, 2, 1, 0, 2])
    # result = generate_compressed(text, d)
    # [byte_to_bits(byte) for byte in result]
    #
    #
    # text = bytes([1, 2, 1, 0])
    # result = generate_compressed(text, d)
    #
    #
    # d1 = {0: "0", 1: "10", 2: "11"}
    # text = bytes([1, 2, 1, 0])
    # d2 = make_freq_dict(bytes([1, 2, 1, 0]))
    # treee = huffman_tree(d2)
    # i = get_codes(treee)
    # assert d1 == get_codes(treee)

    left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    tree = HuffmanNode(None, left, right)
    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    improve_tree(tree, freq)
    avg_length(tree, freq)
