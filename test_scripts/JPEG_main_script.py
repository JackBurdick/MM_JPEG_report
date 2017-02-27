import numpy as np
from skimage import io
from matplotlib import pyplot as plt
input_image_asTXT_path = './image_input.txt'
# huffan_lookup_txt_path = './huffman_tb_dx_input.txt'

# LOSSLESS CODECS
CURCASE = 0

# NOTE: Hardcoded values. this could be modified to be evaluated at run time
# img_height = 16
# img_width = 16

# initialize empty matrix and lookup dict

huffman_dict = {}

def create_image_matrix(input_image_asTXT_path, img_height, img_width):
    original_image = np.zeros((img_width, img_height))
    # fill empty matrix with specified values
    with open(input_image_asTXT_path) as inVals:
        img_row = []
        row_num = 0
        for line in inVals:
            line = line.strip("\n")
            line_list = line.split(" ")
            line_l_clean = [int(val.strip(" ")) for val in line_list if val != ""]
            # print(line_l_clean)
            original_image[row_num] = line_l_clean
            row_num += 1
    return original_image

original_image = create_image_matrix(input_image_asTXT_path, 16, 16)

def create_bin_dict(MAX_LEN):
    global huffman_dict
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
        created_dict[str(val)] = cur_string
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
        created_dict[str(val)] = cur_string
        # print("-val: ", val, "string: \t", cur_string)
    huffman_dict = created_dict

create_bin_dict(22)



# generate prediction based
# input ABC
# return val_not
def prediction_generator(cur_val, case, values):
    A = values[0]
    B = values[1]
    C = values[2]

    float_flag = False

    if(case == 0):
        val_not = A
    elif(case == 1):
        val_not = B
    elif(case == 2):
        val_not = C
    elif(case == 3):
        val_not = A + B - C
    elif(case == 4):
        val_not = A + ((B - C)/2)
        float_flag = True
    elif(case == 5):
        val_not = B + ((A - C)/2)
        float_flag = True
    elif(case == 6):
        val_not = (A + B)/2
        float_flag = True
    else:
        print("we didn't plan for this....")


    diff = cur_val - val_not
    if float_flag:
        # pass
        diff_double = diff * 2
        diff = diff_double
    return diff


def get_pred_row_1(row, cur_col):
    # pred = current - (A)left
    A_val = row[cur_col - 1]
    x_val = row[cur_col]
    pred = x_val - A_val
    # DEBUGGING: print statement showing pred calc
    # print("A(", A_val, ")", "-", "cur(", x_val, ")", " = ", pred)
    return pred


# TODO: this could be grabbed by using a transform on the array
def get_pred_col_1(cur_row, cur_col, original_image):
    # pred = current - (B)above
    B_val = original_image[cur_row-1][cur_col]
    x_val = original_image[cur_row][cur_col]
    pred = x_val - B_val
    # DEBUGGING: print statement showing pred calc
    # print("cur(", x_val, ")", "-", "B(", B_val, ")", " = ", pred)
    return pred


# create prediction matrix
# this uses the prediction_generator to get each value
def create_differences_matrix_with_case(original_image, case):

    img_width, img_height = original_image.shape
    # index values
    cur_row = 0
    cur_col = 0

    # create difference matrix
    # TODO: this should be calc, not hard coded
    diff_matrix = np.zeros((img_width, img_height))

    # copy first value
    diff_matrix[cur_row][cur_col] = original_image[cur_row][cur_col]
    # loop rows in the image
    for row in original_image:
        # first row is a special case
        if cur_row == 0:
            cur_col = 0
            for col_val in row:
                # the first value is already copied
                if cur_col != 0:
                    value = get_pred_row_1(row, cur_col)
                    diff_matrix[cur_row][cur_col] = value
                cur_col += 1
        else:
            cur_col = 0
            for col_val in row:
                if cur_col == 0:
                    diff_value = get_pred_col_1(cur_row,
                                                cur_col,
                                                original_image)
                    diff_matrix[cur_row][cur_col] = diff_value
                else:
                    # Main implementation (first row/col have been calculated)
                    # get prediction according to case
                    current_val = original_image[cur_row][cur_col]
                    A = original_image[cur_row][cur_col-1]
                    B = original_image[cur_row-1][cur_col]
                    C = original_image[cur_row-1][cur_col-1]
                    values = [A, B, C]
                    diff_value = prediction_generator(current_val,
                                                      case,
                                                      values)
                    diff_matrix[cur_row][cur_col] = diff_value
                cur_col += 1
        cur_row += 1
    return diff_matrix

diff_matrix = create_differences_matrix_with_case(original_image, CURCASE)
#print(diff_matrix)


def encode_diffMatrix(diff_matrix, huffman_dict):
    # loop through image
    row_ind = 0
    # col_index = 0
    encoded_string = ""
    for row in diff_matrix:
        col_index = 0
        for col_val in row:
            if row_ind == 0 and col_index == 0:
                # hard coded
                encoded_value = '01011000'
                # convert to binary
            else:
                cur_val = diff_matrix[row_ind][col_index]
                encoded_value = huffman_dict[str(int(cur_val))]
            encoded_string += encoded_value
            col_index += 1
        row_ind += 1
    return encoded_string

# encode
encoded_string = encode_diffMatrix(diff_matrix, huffman_dict)
# print(encoded_string)


def decode_first_row_val(cur_index, created_dif_matrix, img_frm_decoded):
    # new_val = left(A) + diff
    cur_diff = created_dif_matrix[0][cur_index]
    A_val = img_frm_decoded[0][cur_index - 1]
    img_val = A_val + cur_diff
    return img_val


def decode_first_col_val(cur_row_index, cur_col_index,
                         created_dif_matrix, img_frm_decoded):
    # new_val = above(B) + diff
    cur_diff = created_dif_matrix[cur_row_index][cur_col_index]
    B_val = img_frm_decoded[cur_row_index - 1][cur_col_index]
    img_val = B_val + cur_diff
    return img_val


def decode_val_from_case(case,
                         row_index, col_index,
                         created_dif_matrix,
                         img_frm_decoded):
    cur_diff = created_dif_matrix[row_index][col_index]

    A = img_frm_decoded[row_index][col_index-1]
    B = img_frm_decoded[row_index-1][col_index]
    C = img_frm_decoded[row_index-1][col_index-1]

    float_flag = False
    if(case == 0):
        cmp_val = A
    elif(case == 1):
        cmp_val = B
    elif(case == 2):
        cmp_val = C
    elif(case == 3):
        cmp_val = A + B - C
    elif(case == 4):
        cmp_val = A + ((B - C)/2)
        float_flag = True
    elif(case == 5):
        cmp_val = B + ((A - C)/2)
        float_flag = True
    elif(case == 6):
        cmp_val = (A + B)/2
        float_flag = True
    else:
        print("we didn't plan for this....")

    if float_flag:
        cur_diff /= 2

    new_val = cmp_val + cur_diff
    # print(new_val)
    return new_val


def decode_binaryString_to_diff_matrix(encoded_string,
                                       img_width,
                                       img_height,
                                       huffman_dict_rev,
                                       case):
    created_dif_matrix = np.zeros((img_width, img_height))
    i = 0
    cur_seq = ""
    first_num = encoded_string[:8]
    # convert to int
    first_img_val = int(first_num, 2)
    val_list = []
    val_list.append(first_img_val)
    img_values = encoded_string[8:]
    for bin_seq in img_values:
        cur_seq += bin_seq
        if cur_seq in huffman_dict_rev:
            val_int = huffman_dict_rev[cur_seq]
            # DEBUGGING: print statement to ensure conversion
            # print("#", i, ": ", cur_seq, " = ", val_int)
            val_list.append(int(val_int))
            cur_seq = ""
            i += 1

    print(val_list)
    cur_col_index = 0

    # recreate diff matrix
    img_row_index = 0
    img_col_index = 0
    for val in val_list:
        if img_col_index == img_width:
            img_row_index += 1
            img_col_index = 0
        created_dif_matrix[img_row_index][img_col_index % img_width] = val
        img_col_index += 1

    # Show that we've recreated the difference matrix
    # print("CREATED DIF MATRIX")
    # print(created_dif_matrix)

    # create first row in recreated img
    img_frm_decoded = np.zeros((img_width, img_height))
    img_frm_decoded[0][0] = created_dif_matrix[0][0]

    # loop through each row
    row_index = 0
    for diff_row in created_dif_matrix:
        col_index = 0
        # handle special case for first row
        if row_index == 0:
            # start at 1 since the first value is already handled
            col_index += 1
            while col_index < img_width:
                val = decode_first_row_val(col_index,
                                           created_dif_matrix,
                                           img_frm_decoded)
                img_frm_decoded[row_index][col_index] = val
                col_index += 1
        else:
            # print(col_index)
            while col_index < img_width:
                if col_index == 0:
                    # handle special case first column value
                    val = decode_first_col_val(row_index, col_index,
                                               created_dif_matrix,
                                               img_frm_decoded)
                    img_frm_decoded[row_index][col_index] = val
                    col_index += 1
                else:
                    # Main implementation
                    val = decode_val_from_case(case,
                                               row_index, col_index,
                                               created_dif_matrix,
                                               img_frm_decoded)
                    img_frm_decoded[row_index][col_index] = val
                    col_index += 1
        row_index += 1

    # print("CREATED IMAGE")
    # print(img_frm_decoded)
    return img_frm_decoded


# create single image
def create_side_by_side_figure(original_image, recreated_img, CURCASE):
    # plt.figure(figsize=(6, 3))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original")
    # plt.subplot(1, 2, 2)
    axes[1].imshow(recreated_img, cmap='gray')
    title = "Case: " + str(CURCASE)
    axes[1].set_title(title)
    plt.show()

# decode
# reverse mapping in python:
# https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
huffman_dict_rev = {val: key for key, val in huffman_dict.items()}
img_width, img_height = original_image.shape
recreated_img = decode_binaryString_to_diff_matrix(encoded_string,
                                                   img_width,
                                                   img_height,
                                                   huffman_dict_rev,
                                                   CURCASE)

print(original_image)
print("recreated")
print(recreated_img)
create_side_by_side_figure(original_image, recreated_img, CURCASE)
