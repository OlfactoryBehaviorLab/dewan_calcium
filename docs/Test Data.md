
# Combined Data
## Type: Pandas DataFrame
### Size: C x 90 x N
 - ***N*** is the number of trials (usually 264)
 - C is the number of cells (can be whatever we want)
 - The dataframe uses a MultiIndex with two levels to index Cells and Trial Names
	 - Effectively there will be C * N columns, but if we index by cell n it will return just the 264 columns associated with that cell
### Individual Column Description
- Each column is an individual trial
- Each column has 90 rows representing fluorescence values from different periods
	- *Baseline*: Rows 0-34
	- *Odor Response*: Rows 35-55
	- *Post Time*: Rows 55-90
### DataFrame Header (column names)
- The header holds the name of the odorant presented during that trial
- There are 20 odorants and 2 non-odor stimulants for a total of 22 stimuli
		Each Experiment comprises of 5 groups of 4 odorants. The odorants are in the format ***\[modifier\]***-***\[class\]*** e.g. 4-AL; or 1000-OL
		The *odor set* for an experiment consists of the 20 permutations of the modifiers and classes
	- Identity Experiment:
		- Modifiers: \[4, 5, 6, 7]
		- Classes: \[AL, OL, ATE, AMINE, ONE]
	- Concentration Experiment:
		- Modifiers: \[1, 10, 100, 1000]
		- Classes: \[AL, OL, ATE, AMINE, ONE]
	- All Experiments: \[MO, Buzzer]
- The full list of 22 stimuli, is presented 12 times each
	- The total dataset will consist of 264 trials
- The list of 264 odorants/stimuli will then be set as the df header

### Fluorescence Values
- Floating point numbers with the following stats across all cells, but each cell should hypothetically be the same:
	- mean: 0.1168218
	- std: .02726936
	- min: 0.000000
	- 25%: 0.1077778
	- 50%: 0.1146159
	- 75%: 0.1204292
	- max: 0.5744231  


## Possible Fluorescence Patterns
> The patterns exist within the 90 frames per trial. Each trial could have a different below pattern.
1) On Time Excitatory: The average fluorescent values within the *odor* period are higher than those in the *baseline* period
2) On Time Inhibitory: The average fluorescent values within the *baseline* period are higher than those in the *odor* period
3) Latent Excitatory: The average fluorescent values within the *post* period are higher than those in the *baseline* period. There is very little change during the *odor* period. There may be some gradual increasing towards the end of the *odor* period leading into the *post* period
4) Latent Inhibitory: The average fluorescent values within the *baseline* period are higher than those in the *post* period. There is very little change during the *odor* period. There may be some gradual decreasing towards the end of the *odor* period leading into the *post* period

## Generating Test Datasets
- The generation script should accept an excel sheet with the following values:
	- Each row is a cell
		- index_col = 0
	- Each column is an odor
		- header = 0
	- Each cell will contain a number from 0-4 representing the type of response the cell will have to the representative odorant:
	0) No Response
	1) On Time Excitatory
	2) On Time Inhibitory
	3) Latent Excitatory
	4) Latent Inhibitory