# DeTopS_Iso
Isolated version of new measures to implement into DeTopS
VS project for testing small toy examples. 
Link to DeTopS: https://github.com/davidbosc/DeTopS

## Toy Example Details

Consider the following families of sets, where each set contains a group of objects.  Each object, which in this case are simple graphs, can be described by a feature vector where the first element of the vector is the number of vertices in the graph, and the second element of the vector is the number of edges.

![img](https://i.imgur.com/OkYXySc.png)

We'll call the first subset X1, and the second X2, where X is the name of the family of sets.  By applying the function Φ to each set, we can obtain the following object descriptions for each set: 

Φ(A1) = { (2, 1), (3, 3), (3, 2) }

Φ(A2) = { (1, 0), (3, 2), (2, 1) }

Φ(B1) = { (2, 1), (3, 3), (3, 0) }

Φ(B2) = { (2, 1), (3, 2), (4, 3) }

Φ(C1) = { (4, 3), (3, 0), (3, 3) }

Φ(C2) = { (3, 1), (3, 2), (4, 4) }

Next, we'll apply a metric on both families of sets.  The d-iterated pseudometric takes 2 sets of sets as imputs, and applies an embedded metric on those sets.  By using the analog of the [Jaccard Distance](https://en.wikipedia.org/wiki/Jaccard_index) extended to work with set descriptions, we obtain:

Δ<sub>d<sub>J</sub></sub>(A,B) = 1.2

//TODO: work out solutions, add work done by hand, 

Δ<sub>d<sub>J</sub></sub>(A,C) = ???

Δ<sub>d<sub>J</sub></sub>(B,C) = ???

Since d(A,) < d(A,) we can say A is descriptively nearer to  than .

### Toy Example Work

