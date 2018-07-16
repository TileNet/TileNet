**TileNet** is a new neural network architecture that exhibits **excellent generalization** ability on algorithmic tasks, like numeral system conversion , and bubble sorting. On our 9-to-10 numeral system conversion demo task, we successfully generalize to **test set of 2000-digit** numbers, with a **training set of only 5-digit** numbers. We envision this architecture could be extended to be used on other kind of tasks, like NLP tasks, e.g. machine translation.

Table of Contents
=================
  * [Prerequisite](#prerequisite)
  * [Introduction](#introduction)
  * [Contributions](#contributions)
  * [Quickstart](#quickstart)


## Prerequisite

 * pytorch


## Introduction

TileNet gets its name from a mathematical concept called [Tessellation](https://en.wikipedia.org/wiki/Tessellation), which tiles geometric shapes periodically to form specific patterns. The self-resemblance in nature is embodied in the network architecture, as shown in the picture below. The initial motive of designing this architecture is drawn by the insight of specific tasks, like numeral system conversion or bubble sorting,  which practically divides the original problem of `size N` into a smaller one of `size N-1` and a trivial one of `size 1`, and then conquers them.  

To explain the architecture, let's take a 10-to-16 numeral system conversion problem as an example. We have a training example: *100*(Dec) and *64*(Hex). We use *X_i* to denote the source number, *100*, that means, *X_0* for *1*, *X_1* for *0*, and *X_2* for *0*. And we use *Y_i* to denote the target number, *64*, that means, *Y_0* for *6* ad *Y_1* for *4*. Note that we use padding token(*PAD*) to mark the initial condition and the terminal condition, which actually acts like the [Identity Element](https://en.wikipedia.org/wiki/Identity_element) in algebraic terms.

We tile the network layer by layer. On each layer, the *X_i* are used as input, with an initial padding token, and it spews out an element of *Y_i* - Yep, reverse order exactly, see the symmetry of the architecture. We repeat the layer tiling until we encounter the padding token.


```
               PAD         PAD                 PAD         PAD      
                ^           ^                   ^           ^        
                |           |                   |           |                                      
   	    +-------+   +-------+           +-------+   +-------+   
            |  TNN  |   |  TNN  |           |  TNN  |   |  TNN  |
    PAD --->|       |-->|       |--> ... -->|       |-->|       |--> PAD  
            |  Cell |   |  Cell |           |  Cell |   |  Cell |
            +-------+   +-------+           +-------+   +-------+
                ^           ^                   ^           ^
                |           |                   |           |                     
            +-------+   +-------+           +-------+   +-------+   
            |  TNN  |   |  TNN  |           |  TNN  |   |  TNN  |
    PAD --->|       |-->|       |--> ... -->|       |-->|       |--> Y_0   
            |  Cell |   |  Cell |           |  Cell |   |  Cell |
            +-------+   +-------+           +-------+   +-------+
                ^           ^                   ^           ^
                |           |                   |           |                     
            +-------+   +-------+           +-------+   +-------+   
            |  TNN  |   |  TNN  |           |  TNN  |   |  TNN  |
    PAD --->|       |-->|       |--> ... -->|       |-->|       |--> Y_1
            |  Cell |   |  Cell |           |  Cell |   |  Cell |
            +-------+   +-------+           +-------+   +-------+
                ^           ^                   ^           ^
                |           |                   |           |                      
                .           .                   .           .         .
                .           .                   .           . 
                ^           ^                   ^           ^    
                |           |                   |           | 
 	    +-------+   +-------+           +-------+   +-------+   
            |  TNN  |   |  TNN  |           |  TNN  |   |  TNN  |
    PAD --->|       |-->|       |--> ... -->|       |-->|       |--> Y_(m-1)
            |  Cell |   |  Cell |           |  Cell |   |  Cell |
            +-------+   +-------+           +-------+   +-------+
                ^           ^                   ^           ^ 
                |           |                   |           |        
            +-------+   +-------+           +-------+   +-------+
            |  TNN  |   |  TNN  |           |  TNN  |   |  TNN  |
    PAD --->|       |-->|       |--> ... -->|       |-->|       |--> Y_m
            |  Cell |   |  Cell |           |  Cell |   |  Cell |
            +-------+   +-------+           +-------+   +-------+
                ^           ^                   ^           ^
                |           |                   |           |
               X_0         X_1       ...       X_(n-1)     X_n
```

## Contributions

 * With the proper use of the activation function, we effectively applied the concept of fixed-point attractor from Dynamical System theory to TileNet, efficiently preventing accumulation of errors when evaluating on extremely long sequences. Per the experiment, we achieved 100% accuracy even when testing sequences are 400 times longer than training examples.
 * We develped a [Haskell prototype](https://github.com/fracting/tnn_haskell) that is isomorphic to TileNet, successfully demonstrating and explaining the data flow and latent space of Deep Learning model from the point of view of pure functional programming, which might well inspire the future extension of TileNet.

## Quickstart

**NOTE** Now our code contains only the numeral system conversion task(base 9 to base 10) as a demo. Refactor work to extend the code to support user-specified numeral system conversion and other tasks is still under construction.

1. Run the `create_dataset.py` script, which will generate the `data/9-10.txt`, including a 4000-examples training set, 1000-examples validation set, and a 2000-examples test  set.
2. Run the `tnn.py` to start training. The training incorporates random noise, so we validate on a random set on both train set and validation set on each iteration. The training will stop when the accuracy equals to `1.0`, as shown below:

```
training_set size:  4008
75m 20s (- 232m 9s) (9800 24%) average loss 0.004535

> 25
= 23
< ___23

> 43872
= 29144
< 29144

> 43872
= 29144
< 29144

Evaluated on training set loss 0.000000 accuracy 1.000000

> 20474
= 13513
< 13513

> 1
= 1
< ____1

> 1328
= 998
< __998

Evaluated on validation set loss 0.000000 accuracy 1.000000

Stop training
```

The `>` marks the source number, the `<` marks the inferenced number from the model, and the `=` marks the ground truth.

3. After the training is finished, it automatically runs the test set, as shown below:

```
Evaluate on testing_set
> 0
= 0
< _____________________________0

> 84
= 76
< ____________________________76

> 246
= 204
< ___________________________204

> 5315
= 3902
< __________________________3902

> 42314
= 27958
< _________________________27958

> 534551
= 318295
< ________________________318295

> 6177767
= 3299353
< _______________________3299353

```

