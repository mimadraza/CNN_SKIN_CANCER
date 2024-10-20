import numpy as np


class ConvulationalLayer:
    
    filters = []

    def __init__(self,num_filter,filter_size,stride,num_channels):
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.stride = stride
        self.num_channels = num_channels

        #initializing filters
        for i in range(self.num_filter):
            self.filters.append(np.random.random((self.filter_size,self.filter_size,self.num_channels)))

    def forward_prop(self,input):

        result = []

        try:
            padding = int(self.padding_calc(input))
            input = np.pad(input, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

            output_height = (input.shape[0] - self.filter_size) // self.stride + 1
            output_width = (input.shape[1] - self.filter_size) // self.stride + 1

            for filter in self.filters:
                output = np.zeros((input.shape[0],input.shape[1]))
                for i in range(output_height):
                    for j in range(output_width):

                        if i * self.stride + self.filter_size >= input.shape[0]:
                            break
                        if j * self.stride + self.filter_size >= input.shape[1]:
                            break

                        window = input[
                            i * self.stride : i * self.stride + self.filter_size,
                            j * self.stride : j * self.stride + self.filter_size,
                            :
                        ]
                        output[i,j] = np.sum(window*filter)
                result.append(output)

            result = np.stack(result, axis=-1)
            return result
        except IndexError:
            print("The matrix is incompatible")


    def padding_calc(self,input):
        return np.floor((((input.shape[0] - 1)*self.stride) - input.shape[0] + self.filter_size )/2)
