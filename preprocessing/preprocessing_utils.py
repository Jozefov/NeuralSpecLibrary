import torch
import warnings


def spectrum_preparation_double(spectrum, intensity_power, output_size, operation):
    # get spectrum object and return array of specific size for prediction
    # spectrum is in shape tuple[tuple[2]]
    spectrum_output = torch.zeros(1, output_size)

    for position, intensity in spectrum:
        if position >= output_size:
            spectrum_output[0][output_size - 1] = intensity
            continue
        spectrum_output[0][int(position)] = intensity

    if operation == "pow":
        spectrum_output = torch.pow(spectrum_output, intensity_power)
    elif operation == "log":
        spectrum_output = spectrum_output + 1
        spectrum_output = torch.log(spectrum_output)
    else:
        spectrum_output = spectrum_output

    return spectrum_output.type(torch.float64)


def one_hot_encoding(label, num_labels):
    # make one hot encoding for one instance
    # args
    # label: int, position in one hot vector
    # num_label = int, how many groups exist
    # return: torch tensor
    tmp_zeroes = torch.zeros(num_labels)

    if type(label) is bool:
        tmp_zeroes[0] = label
        return tmp_zeroes
    if label >= num_labels:
        tmp_zeroes[num_labels - 1] = float(1)
        warnings.warn("Number of group is greater than one hot dimension representation")
        return tmp_zeroes
    elif label < 0:
        tmp_zeroes[0] = float(1)
        return tmp_zeroes
    else:
        tmp_zeroes[label] = float(1)
    return tmp_zeroes

