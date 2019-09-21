# The Pivot Analysis

The implementation of paper Yao fu, Hao Zhou, Jiaze Chen, and Lei Li, Rethinking Text Attribute Transfer: A Lexical Analysis. INLG 2019. 

# Run it 

```bash
python main.py --dataset=yelp --pivot_thres_cnt=1 --prec_thres=0.5 --recl_thres=0.0
```

and the outputs would something like:

```bash
...
Pivot word discovery:
class 0, 4929 pivots, pivot recall: 0.3348
class 1, 4129 pivots, pivot recall: 0.3435
...
Pivot classifier:
train accuracy: 0.8401
dev accuracy: 0.8313
test accuracy: 0.8333
...
output stored in
../outputs/yelp_1.pivot
```

Parameters tunning:

`prec_thres` gives the confidence of how a word may determine the classification. To find strong pivot words, increase this parameter (e.g. [0.7, 1.0]). To achieve better classification performance, decrease this parameter (e.g. [0.5, 0.7])

`recl_thres` and `pivot_thres_cnt` prevents overfitting on single words. To increase confidence of the pivot words, increase them; to increase classification performance, decrease them.  
