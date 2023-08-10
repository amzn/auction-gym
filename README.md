## AuctionGym: Simulating Online Advertising Auctions

This repository contains the source code for AuctionGym: a simulation environment that enables reproducible offline evaluation of bandit and reinforcement learning approaches to ad allocation and bidding in online advertising auctions.

AuctionGym was released in the context of our ["Off-Policy Learning to Bid with AuctionGym"](https://dl.acm.org/doi/10.1145/3580305.3599877) publication in the Applied Data Science Track of the [2023 ACM SIGKDD Conference](https://kdd.org/kdd2023/).
An [earlier version of our work](https://www.amazon.science/publications/learning-to-bid-with-auctiongym) was presented at the [AdKDD '22 workshop](https://www.adkdd.org/), where it received a Best Paper Award.

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

This section provides instructions to reproduce the results reported in our paper.

We provide a script that takes as input a configuration file detailing the environment and bidders (in JSON format), and outputs raw logged metrics over repeated auction rounds in .csv-files, along with visualisations.
To reproduce the results for truthful bidders in a second-price auction reported in Fig. 1 in the paper, run:

```
python src/main.py config/SP_Oracle.json
```

A `results`-directory will be created, with a subdirectory per configuration file that was ran. This subdirectory will contain .csv-files with raw metrics, and .pdf-files with general visualisations.
Other configuration files will generate results for other environments, and other bidder behaviour.
See [configuration](CONFIG.md) for more detail on the structure of the configuration files.



## Citing


Please cite the [accompanying research paper](https://dl.acm.org/doi/10.1145/3580305.3599877) if you use AuctionGym in your work:

```BibTeX
	@inproceedings{10.1145/3580305.3599877,
		author = {Jeunen, Olivier and Murphy, Sean and Allison, Ben},
		title = {Off-Policy Learning-to-Bid with AuctionGym},
		year = {2023},
		isbn = {9798400701030},
		publisher = {Association for Computing Machinery},
		address = {New York, NY, USA},
		url = {https://doi.org/10.1145/3580305.3599877},
		doi = {10.1145/3580305.3599877},
		booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
		pages = {4219–4228},
		numpages = {10},
		keywords = {online advertising, counterfactual inference, off-policy learning},
		location = {Long Beach, CA, USA},
		series = {KDD '23}
	}
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
