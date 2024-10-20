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
            padding = self.padding_calc(input)
            input_padded= np.pad(input, ((int(padding[0]), int(padding[0])), (int(padding[1]), int(padding[1])), (0, 0)), mode='constant', constant_values=0)

            for filter in self.filters:
                output = np.zeros((input.shape[0],input.shape[1]))
                for i in range(input_padded.shape[0]):
                    for j in range(input_padded.shape[1]):

                        if i * self.stride + self.filter_size >= input_padded.shape[0]:
                            continue 

                        if j * self.stride + self.filter_size >= input_padded.shape[1]:
                            continue 

                        window = input_padded[
                        i * self.stride : i * self.stride + self.filter_size,
                        j * self.stride : j * self.stride + self.filter_size,
                        :
                        ]
                        output[i, j] = np.sum(window * filter)

                result.append(output)

            result = np.stack(result, axis=-1)
            return result
        except IndexError:
            print("The matrix is incompatible")


    def padding_calc(self,input):
        return (np.floor(((self.stride*(input.shape[0] - 1)) - input.shape[0] + self.filter_size)/2) , np.floor(((self.stride*(input.shape[1] - 1)) - input.shape[1] + self.filter_size)/2))
