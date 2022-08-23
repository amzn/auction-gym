## AuctionGym: Simulating Online Advertising Auctions

This repository contains the source code for AuctionGym: a simulation environment that enables reproducible offline evaluation of bandit and reinforcement learning approaches to ad allocation and bidding in online advertising auctions.
A [research paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym) accompanying this repository was accepted as a contribution to the [AdKDD '22 workshop](https://www.adkdd.org/), co-located with the [2022 ACM SIGKDD Conference](https://kdd.org/kdd2022/index.html), and received a Best Paper Award.

Offline evaluation of "learning to bid" approaches is not straightforward, because of multiple reasons:
(1) observational data suffers from unobserved confounding and experimental data with broad interventions is costly to obtain,
(2) offline experiments suffer from Goodhart's Law: " *when a measure becomes a target, it ceases to be a good measure* ", and 
(3) at the time of writing and to the best of our knowledge -- there are no publicly available datasets to researchers that can be used for this purpose.
As a result, reliable and reproducible validation of novel "learning to bid" methods is hindered, and so is open scientific progress in this field.

AuctionGym aims to mitigate this problem, by providing a unified framework that practitioners and research can use to benchmark novel methods and gain insights into their inner workings.


## Getting Started

We provide two introductory and exploratory notebooks. To open them, run `jupyter notebook` in the main directory and navigate to `src`.

" *Getting Started with AuctionGym (1. Effects of Competition)* " simulates second-price auctions with varying levels of competition, visualising the effects on advertiser welfare and surplus, and revenue for the auctioneer.
Analogosuly, " *Getting Started with AuctionGym (2. Effects of Bid Shading)* " simulates first-price auctions where bidders bid truthfully vs. when they shade their bids in a value-based manner.


## Reproducing Research Results

This section provides instructions to reproduce the results reported in our AdKDD paper.

We provide a script that takes as input a configuration file detailing the environment and bidders (in JSON format), and outputs raw logged metrics over repeated auction rounds in .csv-files, along with visualisations.
To reproduce the results for truthful bidders in a second-price auction reported in Fig. 1 in the paper, run:

```
python src/main.py config/SP_Oracle.json
```

A `results`-directory will be created, with a subdirectory per configuration file that was ran. This subdirectory will contain .csv-files with raw metrics, and .pdf-files with general visualisations.
Other configuration files will generate results for other environments, and other bidder behaviour.
See [configuration](CONFIG.md) for more detail on the structure of the configuration files.



## Citing


Please cite the [accompanying research paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym) if you use AuctionGym in your work:

```BibTeX
    @inproceedings{Jeunen2022_AuctionGym,
      author = {Jeunen, Olivier and Murphy, Sean and Allison, Ben},
      title = {Learning to Bid with AuctionGym},
      booktitle = {Proc. of the AdKDD Workshop at the 28th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
      series = {AdKDD '22},
      year = {2022}
    }
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

