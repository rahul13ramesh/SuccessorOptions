# Successor Options

* Code for the paper [Successor Options](https://www.ijcai.org/proceedings/2019/0458.pdf)
* Generates options that navigate to *prototypical states* of well
  connected regions. These states are typically representatives of
  well-connected regions.
* Successor options implicitly clusters the state space into different
  regions and has each option navigating to a different region.

![Successor Options overview](./assets/intro.png)

* To install all required packages run 

```
pip install -r requirements.txt
```


Files and their utilities
=========================

| File                    | Utility                                |
|-------------------------|----------------------------------------|
| diffTime                | Plots where the agent spends its time  |
| eigenPolicies           | Get eigen options                      |
| getPerformance          | Performance of the agent               |
| laplacian               | Class defining eigen-options           |
| successorPolicies       | Get SR-options                         |
| successor               | Class defining SR-options              |
| visualize               | Looks at the SR and Eigen-values       |
| support/                | Helper functions for SR/Eigen options  |
| randomTests/            | Some plots to test/validate ideas      |
| env/                    | Gridworld environment wrapper          |
| data/                   | Folder to store policies               |


