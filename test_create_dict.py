

# 0,1
# 1,00
# -1,011
# 2,0100
def create_bin_dict(MAX_LEN):
    created_dict = {}
    created_dict['0'] = '1'
    max_len = MAX_LEN
    # add positives
    pos_end_string = '00'
    val = 0
    while val <= max_len:
        cur_string = ''
        i = 0
        while i < val:
            cur_string += '01'
            i += 1
        val += 1
        cur_string += pos_end_string
        created_dict[val] = cur_string
        # print("val: ", val, "string: \t", cur_string)

    # add negatives
    neg_end_string = '1'
    val = -0
    while val >= -max_len:
        cur_string = ''
        i = 0
        while i >= val:
            cur_string += '01'
            i -= 1
        val -= 1
        cur_string += neg_end_string
        created_dict[val] = cur_string
        # print("-val: ", val, "string: \t", cur_string)
    return created_dict

jack = create_bin_dict(22)
print(jack)
