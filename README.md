# stDoG - Structure and Dynamics on Graphs (Beta)

stDoG is a Tensorflow Python module for efficiently simulating phase oscillators (the Kuramoto model) on large heterogeneous networks. It provides an implementation for integrating differential equations using TensorFlow, making simulations suitable to be performed on GPUs.

## Install

To install stDoG, download the package and type at the command line: 

```
python install setup.py
```

## Example



```python
import numpy as np
import igraph as ig
from stdog.utils.misc import ig2sparse  #Function to convert igraph format to sparse matrix
from stdog.dynamics.kuramoto import Heuns


#Graph and Kuramoto Parameters
N = 2000
per = 3/N
D = 1.0 
num_couplings = 20

G  = ig.Graph.Erdos_Renyi(N, per)
N = G.vcount()
j = np.arange(1,N+1)

omegas = D*np.tan((j*np.pi)/N - ((N+1.)*np.pi)/(2.0*N)  )
adj = ig2sparse(G)

couplings = np.linspace(0.0,4.,num_couplings)
phases = np.random.uniform(-np.pi,np.pi,N, )
phases =  np.array([
    np.random.uniform(-np.pi,np.pi,N)
    for i_l in range(num_couplings)

],dtype=np.float32)


# Simulation parameters
precision =32
num_temps = 2000
dt = 0.1
total_time = dt*num_temps
transient = False

# stDoG code
heuns_0 = Heuns(
    adj,
    phases,
    omegas, 
    couplings,
    total_time,
    dt,
    device="/cpu:0",
    precision=precision,
    transient = transient,
    use_while=False

)

heuns_0.run()

transient=True
heuns_0.transient = transient
heuns_0.total_time = total_time
heuns_0.run()
order_parameter_list = heuns_0.order_parameter_list

```

## About




## References

[Thomas Peron](https://tkdmperon.github.io/), [Bruno Messias](http://brunomessias.com/), Angélica S. Mata, [Francisco A. Rodrigues](http://conteudo.icmc.usp.br/pessoas/francisco/), and [Yamir Moreno](http://cosnet.bifi.es/people/yamir-moreno/). On the onset of synchronization of Kuramoto oscillators in scale-free networks. [arXiv:1905.02256](https://arxiv.org/abs/1905.02256) (2019).

## Acknowledgements

This work has been supported also by FAPESP grants  11/50761-2  and  2015/22308-2.   Research  carriedout using the computational resources of the Center forMathematical  Sciences  Applied  to  Industry  (CeMEAI)funded by FAPESP (grant 2013/07375-0).
 
### Responsible authors

[@devmessias](https://github.com/devmessias), [@tkdmperon](https://github.com/tkdmperon)
