# Semantic Segmentation
### Introduction
The goal of this project is to classify every pixel of the image to find the drivable portion using a Fully Convolutional neural network based on VGG16 architecture.

### Architecture

Have used pretrained VGG-16 model which was modified to create fully convolutional neural network by adding 1x1 convolution to final layer which is then connected to 4th layer and 3rd layer using skip connection to improve performance. 
Kernel and Strides values are taken from classroom code .

##### Hyper Parameters

Below are the final parameters values after hit and trial method.
`Epoch : 40`

`Batch Size : 5 `

`Learning Rate : 0.001`

`Keep Probability : 0.8`


##### Loss Values over Epoch

`EPOCH :  0  Loss :  0.820696778339`

`EPOCH :  1  Loss :  0.284700981245`

`EPOCH :  2  Loss :  0.170167423528`

`EPOCH :  3  Loss :  0.157447482234`

`EPOCH :  4  Loss :  0.145678557584`

`EPOCH :  5  Loss :  0.133306355954`

`EPOCH :  6  Loss :  0.11424889377`

`EPOCH :  7  Loss :  0.102359644747`

`EPOCH :  8  Loss :  0.0944054434397`

`EPOCH :  9  Loss :  0.100759724091`

`EPOCH :  10  Loss :  0.0834884996173`

`EPOCH :  11  Loss :  0.0730300499447`

`EPOCH :  12  Loss :  0.0697741241291`

`EPOCH :  13  Loss :  0.0648069519164`

`EPOCH :  14  Loss :  0.0817101982519`

`EPOCH :  15  Loss :  0.0703201437819`

`EPOCH :  16  Loss :  0.0605596138871`

`EPOCH :  17  Loss :  0.0553651421877`

`EPOCH :  18  Loss :  0.0496094484396`

`EPOCH :  19  Loss :  0.0497204409591`

`EPOCH :  20  Loss :  0.0468561938033`

`EPOCH :  21  Loss :  0.0440369737816`

`EPOCH :  22  Loss :  0.0409398253622`

`EPOCH :  23  Loss :  0.0594798678789`

`EPOCH :  24  Loss :  0.0626442739419`

`EPOCH :  25  Loss :  0.0548789497015`

`EPOCH :  26  Loss :  0.0623420660609`

`EPOCH :  27  Loss :  0.0961022931954`

`EPOCH :  28  Loss :  0.0871832781182`

`EPOCH :  29  Loss :  0.0817044183355`

`EPOCH :  30  Loss :  0.0598558587953`

`EPOCH :  31  Loss :  0.0466346137611`

`EPOCH :  32  Loss :  0.044739155163`

`EPOCH :  33  Loss :  0.0498478196176`

`EPOCH :  34  Loss :  0.0619639965609`

`EPOCH :  35  Loss :  0.0445585065232`

`EPOCH :  36  Loss :  0.0379096520422`

`EPOCH :  37  Loss :  0.0354083959045`

`EPOCH :  38  Loss :  0.0331410406928`

`EPOCH :  39  Loss :  0.0313383085846`


### Result
All the result is under folder `runs/1526906391.681863` . Below are few of the picks

![image1](./runs/1526906391.681863/um_000067.png)
![image2](./runs/1526906391.681863/um_000058.png)
![image3](./runs/1526906391.681863/um_000032.png)
![image4](./runs/1526906391.681863/uu_000098.png)
![image5](./runs/1526906391.681863/uu_000093.png)
![image6](./runs/1526906391.681863/um_000090.png)
![image7](./runs/1526906391.681863/umm_000004.png)
![image8](./runs/1526906391.681863/um_000082.png)