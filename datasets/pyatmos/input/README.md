This folder should contain the PyATMOS data set, or symlink to it.
The expected directory structure is this one (output of `ls`):

```
376K  dir_0/
368K  dir_1/
372K  dir_2/
372K  dir_3/
376K  dir_4/
368K  dir_5/
364K  dir_6/
372K  dir_7/
368K  dir_8/
380K  dir_9/
2.2M  Dir_alpha/
 73G  Dir_alpha.tar
1.9K  download.bat*
 19M  pyatmos_summary.csv
```

Each of the `dir_*` folders contains the simulation outputs in a directory whose name is a hash starting with `*`.
For example, `dir_0` looks like this:

```
07c2d2fcb1aeec69dab4306deaefc445/
0c2866e2ca23705aa18723e28d9483c3/
08910a081d70779746f37e9a21288b19/
...
```