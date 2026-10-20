# Threshold-conversion reproduction

Empirical and formal study of the whiteboard claim: a multiclass argmax rule can be
written as a 0.5-threshold binary rule after a conversion vector.

Run:

```bash
bash repro/run.sh
```

The run reads `repro/config.json`, writes figures and `results.json` to `repro/out/`,
and prints every metric to stdout so the run log is self-contained.

Tracks:

- `theory`: exact binary conversion, uniform-simplex closed form, fixed-conversion
  counterexamples, instance-adaptive conversion, temperature invariance.
- `digits`: 10-class logistic regression on sklearn digits (a controlled model).
- `text`: binary SST-2 control, 3-class TweetEval sentiment, 6-class emotion.
- `vision`: ViT-B/16 (1000 classes) on CIFAR-10 test images.
- `conversion_fit`: fit a fixed affine/multinomial conversion to the argmax labels
  and report its test accuracy (the empirical no-go check).
