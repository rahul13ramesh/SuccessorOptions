# Successor Options

* Code for the paper [Successor Options](https://www.ijcai.org/proceedings/2019/0458.pdf)
* Generates options that navigate to *prototypical states* of well
  connected regions. These states are typically representatives of
  well-connected regions.
* Successor options implicitly clusters the state space into different
  regions and has each option navigating to a different region.

![Successor Options overview](./assets/intro.png)

* Install the gym-robotics environments from [gym](https://gym.openai.com/envs/#robotics)
* The code uses the [Stable baselines](https://github.com/hill-a/stable-baselines)
* To get the set of configurations to run the code in:
```
python3 qlearning/main.py -h
```

* Use the successor_collect flag followed by the  successor_learn flag.
  The former builds the successor representation while the latter learns
  the successor options.

