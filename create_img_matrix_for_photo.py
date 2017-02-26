import numpy as np

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

img_matrix = create_image_matrix('./image_input.txt', 16, 16)
print(img_matrix)
