# DeTopS_Iso
Isolated version of new measures to implement into DeTopS
VS project for testing small toy examples. 
Link to DeTopS: https://github.com/davidbosc/DeTopS

## Toy Example Details

Consider the following families of sets, where each set contains a group of objects.  Each object, which in this case are simple graphs, can be described by a feature vector where the first element of the vector is the number of vertices in the graph, and the second element of the vector is the number of edges.

![img](https://i.imgur.com/OkYXySc.png)

We'll call the first subset X<sub>1</sub>, and the second X<sub>2</sub>, where X is the name of the family of sets.  By applying the function Φ to each set, we can obtain the following object descriptions for each set: 

Φ(A<sub>1</sub>) = { (2, 1), (3, 3), (3, 2) }

Φ(A<sub>2</sub>) = { (1, 0), (3, 2), (2, 1) }

Φ(B<sub>1</sub>) = { (2, 1), (3, 3), (3, 0) }

Φ(B<sub>2</sub>) = { (2, 1), (3, 2), (4, 3) }

Φ(C<sub>1</sub>) = { (4, 3), (3, 0), (3, 3) }

Φ(C<sub>2</sub>) = { (3, 1), (3, 2), (4, 4) }

Next, we'll apply a metric on both families of sets.  The d-iterated pseudometric takes 2 sets of sets as imputs, and applies an embedded metric on those sets.  By using the analog of the [Jaccard Distance](https://en.wikipedia.org/wiki/Jaccard_index) extended to work with set descriptions, we obtain:

Δ<sub>d<sub>J</sub></sub>(A,B) = 0.575 

Δ<sub>d<sub>J</sub></sub>(A,C) = 0.85

Δ<sub>d<sub>J</sub></sub>(B,C) = 0.775

Since Δ<sub>d<sub>J</sub></sub>(A,B) < Δ<sub>d<sub>J</sub></sub>(A,C) we can say A is descriptively nearer to B than C.

Also, since Δ<sub>d<sub>J</sub></sub>(A,B) = Δ<sub>d<sub>J</sub></sub>(B,A) < Δ<sub>d<sub>J</sub></sub>(B,C) we can say B is descriptively nearer to A than C.

Lastly, since Δ<sub>d<sub>J</sub></sub>(B,C) < Δ<sub>d<sub>J</sub></sub>(A,C) we can say C is descriptively nearer to B than A.


We're not restricted to using Jaccard here, we can actual use any pseudometric with this d-iterated pseudometric.  By using the analog of the [Hausdorff Distance](https://en.wikipedia.org/wiki/Hausdorff_distance) extended to work with feature descriptions, we obtain the following equation:

![img](https://i.imgur.com/5946xky.png)

Note that the Hausdorff distance takes a metric to perform on the description of features.  Going forward, we'll use the Hamming Distance, treating each feature as a string and matching them on the i-th digits. From here, we can then obtain:

Δ<sub>d<sub>H</sub></sub>(A,B) = 1.25 

Δ<sub>d<sub>H</sub></sub>(A,C) = 2

Δ<sub>d<sub>H</sub></sub>(B,C) = 1.75

Since Δ<sub>d<sub>H</sub></sub>(A,B) < Δ<sub>d<sub>H</sub></sub>(A,C) we can say A is again, descriptively nearer to B than C.

Also, since Δ<sub>d<sub>H</sub></sub>(A,B) = Δ<sub>d<sub>H</sub></sub>(B,A) < Δ<sub>d<sub>H</sub></sub>(B,C) we can say B is descriptively nearer to A than C.

Lastly, since Δ<sub>d<sub>H</sub></sub>(B,C) < Δ<sub>d<sub>H</sub></sub>(A,C) we can say C is descriptively nearer to B than A.

### Toy Example Work

#### Jaccard Distance Calculations per Subset

![img](https://i.imgur.com/SBgDjXu.png)


#### d-iterated pseudometric per Family of Sets with Jaccard

![img](https://i.imgur.com/cYfGs8m.png)

#### Hausdorff Distance (with Hamming Distance) Calculations per Subset

![img](https://i.imgur.com/4ZnQQrS.png)

![img](https://i.imgur.com/jj6ETND.png)

![img](https://i.imgur.com/olGVSCF.png)

![img](https://i.imgur.com/4Vt9FE1.png)

b1c
![img](https://i.imgur.com/cifSv6u.png)

b2c
![img](https://i.imgur.com/SlLvbqu.png)

#### d-iterated pseudometric per Family of Sets with Hausdorff

![img](https://i.imgur.com/JnUVKXD.png)
