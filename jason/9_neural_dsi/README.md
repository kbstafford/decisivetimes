# #9: Exploring a neural decision strength index

The approach described in #9 for a neural decision strength index is less relevant now, but this provides an example of how to fit simple logistic regression decoders that can predict left/right decisions with high accuracy.

## To reproduce

Basic logistic regression and trajectory plots can be computed for many recordings like this:

```bash
for i in {1..20}; do python predict_left_right.py --seed "$i"; done
```
