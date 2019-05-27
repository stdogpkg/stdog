# stDoG - Structure and Dynamics on Graphs (Beta)

stDoG is a Tensorflow Python module for efficiently simulating phase oscillators (the Kuramoto model) on large heterogeneous networks. It provides an implementation for integrating differential equations using TensorFlow, making simulations suitable to be performed on GPUs.

## Install

To install stDoG, download the package and type at the command line: 

```
python install setup.py
```

## Example



```python

from tkuramoto.utils import ig2sparse #Function to convert igraph format to sparse matrix
from tkuramoto.gpu.heuns import Heuns
...
```

## About




## References

[Thomas Peron](https://tkdmperon.github.io/), [Bruno Messias](http://brunomessias.com/), Angélica S. Mata, [Francisco A. Rodrigues](http://conteudo.icmc.usp.br/pessoas/francisco/), and [Yamir Moreno](http://cosnet.bifi.es/people/yamir-moreno/). On the onset of synchronization of Kuramoto oscillators in scale-free networks. [arXiv:1905.02256](https://arxiv.org/abs/1905.02256) (2019).

## Acknowledgements

This work has been supported also by FAPESP grants  11/50761-2  and  2015/22308-2.   Research  carriedout using the computational resources of the Center forMathematical  Sciences  Applied  to  Industry  (CeMEAI)funded by FAPESP (grant 2013/07375-0).
 
### Responsible authors

[@devmessias](https://github.com/devmessias), [@tkdmperon](https://github.com/tkdmperon)
