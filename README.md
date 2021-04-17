#### Final Project Submission for the Digital Audio Signal Processing course at McGill University!

#### Group: Dorian Desblancs, James Mesich

#### Report Abstract:

In this project, we implemented two well known audio restoration techniques. The first was taken from Digital Audio Restoration by Godsill et al. [1] whilst the second was presented by Laurent Oudre in 2015 and 2018 [2] [3]. The former is primarily focused on wow and flutter removal. Both artefacts are due to unwanted pitch variations in a recording, and are restored using an estimate of the pitch variation curve. This technique relies heavily on clever pre-processing and Bayesian probability. While testing on simple signals produced reliable results, the implementation failed to remove defects in more complex audio such as gramophone recordings. The second algorithm relies on identifying and interpolating bursts of noisy audio. This method was designed to remove defects such as clicks and hiss commonly found in vinyl records. Through proper tuning of parameters, strong restoration results were achieved on complex audio recordings.
