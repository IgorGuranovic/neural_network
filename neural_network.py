import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy, pickle

class Residue:
    def __init__(self, name, molar_mass, vdw_volume, hydro, pk_COOH, pk_NH3, aromatic_aliphatic):
        self.name = name
        self.molar_mass = molar_mass
        self.vdw_volume = vdw_volume
        self.hydro = hydro
        self.pk_COOH = pk_COOH
        self.pk_NH3 = pk_NH3
        self.aromatic_aliphatic = aromatic_aliphatic

#hydrophobic = 1, hydrophilic = 0
#aromatic = 0, aliphatic = 1, neither = 0.5

Alanine = Residue("A", 15.03, 67, 1, 2.34, 9.69, 0.5)
Cysteine = Residue("C", 47.10, 86, 0, 1.96, 10.128, 0.5)
Aspartic_Acid = Residue("D", 59.04, 91, 0, 1.88, 9.60, 0.5)
Glutamic_Acid = Residue("E", 73.07, 109, 0, 2.19, 9.67, 0.5)
Phenylalanine = Residue("F", 91.13, 135, 1, 1.83, 9.13, 0)
Glycine = Residue("G", 1.01, 48, 1, 2.34, 9.60, 0.5)
Histidine = Residue("H", 81.10, 118, 0, 1.82, 9.17, 0)
Isoleucine = Residue("I", 57.11, 124, 1, 2.36, 9.60, 1)
Lysine = Residue("K", 72.13, 135, 0, 2.18, 8.95, 0.5)
Leucine = Residue("L", 57.11, 124, 1, 2.36, 9.60, 1)
Methionine = Residue("M", 75.15, 124, 1, 2.28, 9.21, 0.5)
Asparagine = Residue("N", 58.06, 96, 0, 2.02, 8.80, 0.5)
Proline = Residue("P", 42.08, 90, 1, 1.99, 10.60, 0.5)
Glutamine = Residue("Q", 72.09, 114, 0, 2.17, 9.13, 0.5)
Arginine = Residue("R", 100.14, 148, 0, 2.17, 9.04, 0.5)
Serine = Residue("S", 31.03, 73, 0, 2.21, 9.15, 0.5)
Threonine = Residue("T", 45.06, 93, 0, 2.09, 9.10, 0.5)
Valine = Residue("V", 43.09, 105, 1, 2.32, 9.62, 1)
Tryptophan = Residue("W", 130.17, 163, 0, 2.83, 9.39, 0)
Tyrosine = Residue("Y", 107.13, 141, 0, 2.20, 9.11, 0)

amino_acid_properties = [Alanine, Cysteine, Aspartic_Acid, Glutamic_Acid, Phenylalanine, Glycine, Histidine, Isoleucine, Lysine, Leucine, Methionine, Asparagine, Proline, Glutamine, Arginine, Serine, Threonine, Valine, Tryptophan, Tyrosine]

# format: ["number", "letter", sequence, structure, distance in angstroms]
structure = [["34", "H", .9741, 3, 13.1],["66", "E", .8478, 7, 13.2],["67", "W", .9074, 8, 13.7],["128", "H", .9810, 0, 8.6],["129", "H", .9375, 2, 8.9],["224", "D", .9844, 3, 0.],["254", "R", .4652, 4, 5.5],["266", "E", .8505, 6, 12.4]]

def dataFrame(mutants):
    inProteins = []
    for x in range(len(mutants)):
        inProteins.append(copy.deepcopy(structure))
    outProteins=copy.deepcopy(inProteins)

    for protInd, protein in enumerate(inProteins):
        for mutantIndex, mutant in enumerate(mutants):
            if protInd != mutantIndex:
                continue
            for mutatInd, mutation in enumerate(mutant):
                for aminoInd, amino in enumerate(protein):
                    if amino[0] == mutants[mutantIndex][mutatInd][1:-1]:
                        outProteins[protInd][aminoInd][1] = mutants[mutantIndex][mutatInd][-1]

    for protein in outProteins:
        for amino in protein:
            for index, element in enumerate(amino_acid_properties):
                if amino_acid_properties[index].name == amino[1]:
                    amino.append(amino_acid_properties[index].molar_mass)
                    amino.append(amino_acid_properties[index].vdw_volume)
                    amino.append(amino_acid_properties[index].hydro)
                    amino.append(amino_acid_properties[index].pk_COOH)
                    amino.append(amino_acid_properties[index].pk_NH3)
                    amino.append(amino_acid_properties[index].aromatic_aliphatic)
                    del amino[0:2]

    dataframe = np.asarray(outProteins)
    return dataframe

#Choose your directories for the four "pickle" files

with open('D:/autodock_automation/Pickle_files/mutants_training.pickle', 'rb') as f:
    mutants_training = pickle.load(f)

with open('D:/autodock_automation/Pickle_files/mutants_test.pickle', 'rb') as f:
    mutants_test = pickle.load(f)

with open('D:/autodock_automation/Pickle_files/training_labels.pickle', 'rb') as f:
    training_labels = np.asarray(pickle.load(f))

with open('D:/autodock_automation/Pickle_files/test_labels.pickle', 'rb') as f:
    test_labels = np.asarray(pickle.load(f))

training_data = dataFrame(mutants_training)
test_data = dataFrame(mutants_test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8,9)),
    keras.layers.Dense(256, activation ='relu'),
    keras.layers.Dense(256, activation ='relu'),
    keras.layers.Dense(256, activation ='relu'),
    keras.layers.Dense(256, activation ='relu'),
    keras.layers.Dense(16, activation ='relu'),
    keras.layers.Dense(8, activation ='relu'),
    keras.layers.Dense(1, activation='linear')
])

keras.optimizers.Adam(lr=.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', metrics=['mse'])
model.fit(training_data, training_labels, epochs=5, batch_size=32, validation_split=0.15, validation_data=None, verbose=1)
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose = 1, batch_size=32)

print('Test accuracy:', test_acc)