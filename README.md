Notes about the repo:

1. Main training of the NN happens in combinedModel.py
2. The code that runs SciPy's LSQE is in unmixing.py
3. The setup of the NN architecture is in NN.py
4. Converting the results of SciPy's unmixing into visual images is in VisualizeEMs.py
5. Handling of data for training and testing of the NN is in Data.py
6. Extracting the spectra of the endmembers from their images is in endMemberSpecs.py
7. ManualCombination.py includes helper functions for visualizing NN results pixel by pixel (not totally necessary)