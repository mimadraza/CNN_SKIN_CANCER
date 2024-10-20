import numpy as np

class PoolingLayer:

    def __init__(self,stride,filter_size):
        self.stride = stride
        self.filter_size = filter_size

    def forward_pool(self,input):

        result = []

        output_height = int(np.floor(((input.shape[0] - self.filter_size)/self.stride) + 1))
        output_width = int(np.floor(((input.shape[1] - self.filter_size)/self.stride) + 1))

        try:
            for k in range(input.shape[2]):
                output = np.zeros((output_height, output_width))
                for i in range(output_height):
                    for j in range(output_width):
                    # Extract the window from the input for channel k
                        window = input[
                            i * self.stride : i * self.stride + self.filter_size,
                            j * self.stride : j * self.stride + self.filter_size,
                            k
                        ]
                        # Apply max pooling to the window
                        output[i, j] = np.max(window)

                result.append(output)

            result = np.stack(result, axis=-1)
            return result
        except IndexError:
            print("The matrix is incompatible")
