def extract_number_and_remove_digits(input_string):
    # Initialize variables to store the number and the result without digits
    extracted_number = 0
    result_without_digits = ""

    # Iterate through each character in the input string
    for char in input_string:
        # Check if the character is a digit
        if char.isdigit():
            # If it's a digit, add it to the extracted number
            extracted_number = extracted_number * 10 + int(char)
        else:
            # If it's not a digit, add it to the result string without digits
            result_without_digits += char

    return extracted_number, result_without_digits
