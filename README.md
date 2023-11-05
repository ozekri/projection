# Projection onto the exponential cone and its dual (Python implementation)

This repository contains a Python implementation of the projection onto the exponential cone (and its dual). This is the Python version of the Julia implementation of H. Friberg that you can found [there](https://github.com/HFriberg/projection). The method is from his paper [Projection onto the exponential cone:
a univariate root-finding problem](https://docs.mosek.com/whitepapers/expcone-proj.pdf).

A big thank to H. Friberg because all the credit goes to him. I haven't done much other than re-implementing this in Python syntax, and correcting some approximation errors. I just needed this implementation because I’m not working on Julia. Hoping that it can help someone who found himself with the same need.

<p align="center">
<img src="https://github.com/ozekri/projection/blob/main/POCS_Exponential_cone_0.gif" width=50% height=50% alt>
</p>

<em>Gif animation of POCS algorithm between the exponential cone and the plane </em> $z=0$.

## Usage

```python
import proj_exp_cone as pe

v0 = [z,y,x] #point to project (note that x and z are reversed).

vp, vd = pe.projprimalexpcone(v0)
#vp is the projection onto the exponential cone.
#vd is the projection onto its dual.

pe.solutionreport(v0,vp,vd) #returns solution report, with absolute and relative error.
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://github.com/ozekri/projection/blob/main/LICENSE)
